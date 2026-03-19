use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::time::Duration;
use std::sync::{Arc, Mutex};

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use chrono::Utc;
use crafter_core::renderer::{Renderer, TextRenderer};
use crafter_core::{Action, SaveData, Session, SessionConfig, StepResult};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use uuid::Uuid;

const DEFAULT_MODEL: &str = "gpt-4.1-nano";
const DEFAULT_EPISODE_MAX_STEPS: u32 = 2_000;
const DEFAULT_VIEW_RADIUS: u32 = 4;
const DEFAULT_POLICY_PROMPT: &str = "You control an agent in Crafter. Return strict JSON only as {\"actions\":[\"<action_name>\",\"...\"]} with 1-6 actions. Use movement to explore when nothing useful is adjacent. Use 'do' only when facing a useful nearby object or resource. Read the recent action history and avoid repeating unproductive loops.";
const DEFAULT_OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

#[derive(Clone)]
struct AppState {
    inner: Arc<Mutex<Store>>,
}

#[derive(Default)]
struct Store {
    rollouts: HashMap<String, RolloutRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Waypoint {
    description: String,
    #[serde(default)]
    achievement: Option<String>,
    #[serde(default)]
    inventory_requirements: BTreeMap<String, i64>,
    #[serde(default)]
    player_position: Option<(i32, i32)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointRecord {
    checkpoint_id: String,
    rollout_id: String,
    label: Option<String>,
    source: Option<String>,
    actor_ids: Vec<String>,
    created_at: String,
    metadata: Map<String, Value>,
    checkpoint_bytes: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrajectoryItem {
    step_idx: u64,
    actions: Vec<String>,
    reward: f64,
    total_reward: f64,
    env_reward: f64,
    env_total_reward: f64,
    done: bool,
    player_pos: (i32, i32),
    achievements: Vec<String>,
    newly_unlocked: Vec<String>,
    inventory: Value,
    current_waypoint_index: usize,
}

struct RolloutRecord {
    rollout_id: String,
    trace_correlation_id: String,
    trial_id: Option<String>,
    seed: u64,
    session: Session,
    env_config: Map<String, Value>,
    policy_config: Map<String, Value>,
    planner_mode: Option<String>,
    waypoints: Vec<Waypoint>,
    created_at: String,
    started_at: Option<String>,
    completed_at: Option<String>,
    status: String,
    parent_rollout_id: Option<String>,
    total_reward: f64,
    last_reward: f64,
    env_total_reward: f64,
    env_last_reward: f64,
    last_done: bool,
    last_done_reason: Option<String>,
    last_newly_unlocked: Vec<String>,
    trajectory: Vec<TrajectoryItem>,
    completed_waypoints: Vec<Value>,
    current_waypoint_index: usize,
    planner_failure_code: Option<String>,
    last_status_detail: Option<String>,
    decision_turns: Vec<DecisionTurnRecord>,
    checkpoints: BTreeMap<String, CheckpointRecord>,
    terminated: bool,
    inference_url: Option<String>,
    inference_error_count: u32,
    last_inference_error: Option<String>,
    llm_call_count: u32,
    llm_action_batches: Vec<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DecisionTurnRecord {
    turn_index: usize,
    prompt_messages: Vec<Value>,
    assistant_text: String,
    reasoning_text: Option<String>,
    actions: Vec<String>,
    decision_reward: f64,
    reward_before: f64,
    reward_after: f64,
    env_reward_before: f64,
    env_reward_after: f64,
    step_start: u64,
    step_end: u64,
    trainable: bool,
    invalid_parse: bool,
    behavior_version: String,
    behavior_model: String,
    route: String,
    request_id: Option<String>,
    behavior_sequence_logprob: Option<f64>,
    usage: Value,
}

#[derive(Debug, Default, Deserialize)]
struct RolloutEnvSpec {
    #[serde(default)]
    config: Map<String, Value>,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
struct RolloutPolicySpec {
    #[serde(default)]
    config: Map<String, Value>,
}

#[derive(Debug, Deserialize)]
struct RolloutRequest {
    trace_correlation_id: String,
    #[serde(default)]
    trial_id: Option<String>,
    #[serde(default)]
    env: RolloutEnvSpec,
    #[serde(default)]
    policy: RolloutPolicySpec,
    #[serde(default)]
    planner_mode: Option<String>,
    #[serde(default)]
    waypoint_plan_request: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CheckpointBody {
    #[serde(default)]
    checkpoint_id: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    actor_ids: Vec<String>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    rollout_id: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ResumeOverrides {
    #[serde(default)]
    env: Map<String, Value>,
    #[serde(default)]
    env_config: Map<String, Value>,
    #[serde(default)]
    policy: Map<String, Value>,
    #[serde(default)]
    policy_config: Map<String, Value>,
    #[serde(default)]
    segment_steps: Option<u32>,
    #[serde(default)]
    continue_steps: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
struct ResumeBody {
    #[serde(default)]
    rollout_id: Option<String>,
    #[serde(default)]
    checkpoint_id: Option<String>,
    #[serde(default)]
    target_rollout_id: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    overrides: ResumeOverrides,
    #[serde(default)]
    checkpoint_data_base64: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct TaskInfoQuery {
    #[serde(default)]
    seed: Vec<u64>,
    #[serde(default)]
    seeds: Vec<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct TerminateBody {
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    env_id: Option<String>,
}

#[derive(Debug)]
struct AppError {
    status: StatusCode,
    code: String,
    message: String,
}

impl AppError {
    fn new(status: StatusCode, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": self.code,
                "message": self.message,
            })),
        )
            .into_response()
    }
}

type AppResult<T> = Result<T, AppError>;

#[tokio::main]
async fn main() {
    let state = AppState {
        inner: Arc::new(Mutex::new(Store::default())),
    };

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/done", get(done))
        .route("/info", get(info))
        .route("/task_info", get(task_info))
        .route("/rollout", post(rollout))
        .route("/rollouts", post(rollout))
        .route("/rollouts/:rollout_id", get(get_rollout))
        .route(
            "/rollouts/:rollout_id/checkpoints",
            get(list_checkpoints).post(create_checkpoint),
        )
        .route(
            "/rollouts/:rollout_id/checkpoints/:checkpoint_id",
            get(get_checkpoint),
        )
        .route("/rollouts/:rollout_id/resume", post(resume_rollout))
        .route(
            "/rollouts/:rollout_id/checkpoint/dump",
            post(checkpoint_dump_alias),
        )
        .route(
            "/rollout/:rollout_id/checkpoint/dump",
            post(checkpoint_dump_alias),
        )
        .route(
            "/rollouts/:rollout_id/checkpoint/restore",
            post(checkpoint_restore_alias),
        )
        .route(
            "/rollout/:rollout_id/checkpoint/restore",
            post(checkpoint_restore_alias),
        )
        .route("/checkpoint/save", post(checkpoint_save))
        .route("/checkpoint/list", get(checkpoint_list))
        .route("/checkpoint/load", post(checkpoint_load))
        .route("/rollouts/:rollout_id/terminate", post(terminate_rollout))
        .route("/env/CrafterClassic/terminate", post(legacy_terminate))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8903));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("bind listener");
    println!("crafter-rs container listening on http://{}", addr);
    axum::serve(listener, app).await.expect("serve app");
}

async fn root() -> Json<Value> {
    Json(json!({"status": "ok", "service": "crafter_rs_container"}))
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok", "service": "crafter_rs_container"}))
}

async fn done() -> Json<Value> {
    Json(json!({"ok": true, "service": "crafter_rs_container"}))
}

async fn info() -> Json<Value> {
    let mut payload = task_info_payload(None);
    if let Some(obj) = payload.as_object_mut() {
        obj.insert(
            "rubrics".to_string(),
            json!({
                "outcome": {
                    "criteria": [
                        {
                            "id": "long_horizon_progress",
                            "description": "Reward survival, achievement progression, and successful waypoint execution over long horizons.",
                            "weight": 1.0
                        }
                    ]
                }
            }),
        );
        if let Some(task_metadata) = obj
            .get_mut("task_metadata")
            .and_then(Value::as_object_mut)
        {
            task_metadata.insert(
                "stop_routes".to_string(),
                json!(["/rollouts/{rollout_id}/terminate", "/env/CrafterClassic/terminate"]),
            );
        }
    }
    Json(payload)
}

async fn task_info(Query(query): Query<TaskInfoQuery>) -> Json<Value> {
    let mut seeds = query.seed;
    seeds.extend(query.seeds);
    if seeds.is_empty() {
        return Json(task_info_payload(None));
    }
    if seeds.len() == 1 {
        return Json(task_info_payload(seeds.first().copied()));
    }
    Json(Value::Array(
        seeds.into_iter().map(|seed| task_info_payload(Some(seed))).collect(),
    ))
}

async fn rollout(State(state): State<AppState>, Json(request): Json<RolloutRequest>) -> AppResult<Json<Value>> {
    if request.trace_correlation_id.trim().is_empty() {
        return Err(AppError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "missing_trace_correlation_id",
            "trace_correlation_id must be non-empty",
        ));
    }

    let seed = request
        .env
        .seed
        .unwrap_or_else(|| derive_seed(&request.trace_correlation_id));
    let env_config = request.env.config.clone();
    let policy_config = request.policy.config.clone();
    let segment_steps = segment_steps_from_config(&env_config);
    let mut record = RolloutRecord {
        rollout_id: request.trace_correlation_id.clone(),
        trace_correlation_id: request.trace_correlation_id.clone(),
        trial_id: request.trial_id.clone(),
        seed,
        session: Session::new(session_config_from_env(seed, &env_config)),
        env_config,
        policy_config,
        planner_mode: request.planner_mode.clone(),
        waypoints: parse_waypoints(request.waypoint_plan_request.as_ref()),
        created_at: now_iso(),
        started_at: None,
        completed_at: None,
        status: "pending".to_string(),
        parent_rollout_id: None,
        total_reward: 0.0,
        last_reward: 0.0,
        env_total_reward: 0.0,
        env_last_reward: 0.0,
        last_done: false,
        last_done_reason: None,
        last_newly_unlocked: Vec::new(),
        trajectory: Vec::new(),
        completed_waypoints: Vec::new(),
        current_waypoint_index: 0,
        planner_failure_code: None,
        last_status_detail: None,
        decision_turns: Vec::new(),
        checkpoints: BTreeMap::new(),
        terminated: false,
        inference_url: None,
        inference_error_count: 0,
        last_inference_error: None,
        llm_call_count: 0,
        llm_action_batches: Vec::new(),
    };
    update_waypoint_progress(&mut record);
    run_rollout_segment(&mut record, segment_steps).await;

    let response = rollout_response(&record);
    let mut store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    store.rollouts.insert(record.rollout_id.clone(), record);
    Ok(Json(response))
}

async fn get_rollout(State(state): State<AppState>, Path(rollout_id): Path<String>) -> AppResult<Json<Value>> {
    let store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    let record = store
        .rollouts
        .get(&rollout_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
    Ok(Json(rollout_resource(record)))
}

async fn list_checkpoints(State(state): State<AppState>, Path(rollout_id): Path<String>) -> AppResult<Json<Value>> {
    let store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    let record = store
        .rollouts
        .get(&rollout_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
    Ok(Json(json!({
        "rollout_id": rollout_id,
        "checkpoints": record.checkpoints.values().map(checkpoint_descriptor).collect::<Vec<_>>(),
    })))
}

async fn create_checkpoint(
    State(state): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<Option<CheckpointBody>>,
) -> AppResult<(StatusCode, Json<Value>)> {
    let mut store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    let record = store
        .rollouts
        .get_mut(&rollout_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
    let payload = body.unwrap_or(CheckpointBody {
        checkpoint_id: None,
        label: None,
        source: None,
        actor_ids: Vec::new(),
        metadata: Map::new(),
        rollout_id: None,
    });
    let checkpoint = make_checkpoint(record, payload);
    Ok((StatusCode::CREATED, Json(checkpoint_descriptor(&checkpoint))))
}

async fn get_checkpoint(
    State(state): State<AppState>,
    Path((rollout_id, checkpoint_id)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    let store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    let record = store
        .rollouts
        .get(&rollout_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
    let checkpoint = record
        .checkpoints
        .get(&checkpoint_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_checkpoint", format!("checkpoint not found: {}", checkpoint_id)))?;
    Ok(Json(checkpoint_descriptor(checkpoint)))
}

async fn resume_rollout(
    State(state): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<Option<ResumeBody>>,
) -> AppResult<(StatusCode, Json<Value>)> {
    let payload = body.unwrap_or_default();
    let inline_checkpoint_bytes = if let Some(encoded) = payload
        .checkpoint_data_base64
        .as_ref()
        .filter(|value| !value.trim().is_empty())
    {
        BASE64_STANDARD
            .decode(encoded)
            .map_err(|_| AppError::new(StatusCode::BAD_REQUEST, "invalid_checkpoint_data", "checkpoint_data_base64 could not be decoded"))?
    } else {
        Vec::new()
    };
    let mode = payload.mode.as_deref().unwrap_or("new_rollout");
    if mode != "new_rollout" && mode != "in_place" {
        return Err(AppError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_resume_mode",
            format!("invalid resume mode: {}", mode),
        ));
    }

    let segment_steps = payload
        .overrides
        .segment_steps
        .or(payload.overrides.continue_steps)
        .unwrap_or(1);

    if mode == "in_place" {
        let mut target = {
            let mut store = state
                .inner
                .lock()
                .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
            let mut target = store
                .rollouts
                .remove(&rollout_id)
                .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
            let checkpoint_bytes = if inline_checkpoint_bytes.is_empty() {
                let checkpoint_id = payload
                    .checkpoint_id
                    .clone()
                    .or_else(|| target.checkpoints.keys().last().cloned())
                    .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "no_checkpoints_for_rollout", format!("no checkpoints for rollout: {}", rollout_id)))?;
                target
                    .checkpoints
                    .get(&checkpoint_id)
                    .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_checkpoint", format!("checkpoint not found: {}", checkpoint_id)))?
                    .checkpoint_bytes
                    .clone()
            } else {
                inline_checkpoint_bytes.clone()
            };
            let save_data: SaveData = serde_json::from_slice(&checkpoint_bytes).map_err(|_| {
                AppError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_checkpoint_payload",
                    "checkpoint bytes could not be parsed as crafter-rs SaveData",
                )
            })?;
            let mut env_config = target.env_config.clone();
            env_config.extend(payload.overrides.env.clone());
            env_config.extend(payload.overrides.env_config.clone());
            let mut policy_config = target.policy_config.clone();
            policy_config.extend(payload.overrides.policy.clone());
            policy_config.extend(payload.overrides.policy_config.clone());
            target.session = save_data.into_session();
            target.env_config = env_config;
            target.policy_config = policy_config;
            target.status = "running".to_string();
            target.completed_at = None;
            target.terminated = false;
            update_waypoint_progress(&mut target);
            target
        };
        run_rollout_segment(&mut target, segment_steps).await;
        let response = rollout_response(&target);
        let mut store = state
            .inner
            .lock()
            .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
        store.rollouts.insert(rollout_id, target);
        return Ok((StatusCode::ACCEPTED, Json(response)));
    }

    let (
        checkpoint_bytes,
        seed,
        source_env_config,
        source_policy_config,
        planner_mode,
        waypoints,
        parent_rollout_id,
        source_trial_id,
        total_reward,
        env_total_reward,
        source_completed_waypoints,
        source_trajectory,
        source_decision_turns,
        source_inference_url,
        source_inference_error_count,
        source_last_inference_error,
    ) = {
        let store = state
            .inner
            .lock()
            .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
        let source = store
            .rollouts
            .get(&rollout_id)
            .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
        let checkpoint_bytes = if inline_checkpoint_bytes.is_empty() {
            let checkpoint_id = payload
                .checkpoint_id
                .clone()
                .or_else(|| source.checkpoints.keys().last().cloned())
                .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "no_checkpoints_for_rollout", format!("no checkpoints for rollout: {}", rollout_id)))?;
            source
                .checkpoints
                .get(&checkpoint_id)
                .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_checkpoint", format!("checkpoint not found: {}", checkpoint_id)))?
                .checkpoint_bytes
                .clone()
        } else {
            inline_checkpoint_bytes.clone()
        };
        (
            checkpoint_bytes,
            source.seed,
            source.env_config.clone(),
            source.policy_config.clone(),
            source.planner_mode.clone(),
            source.waypoints.clone(),
            source.rollout_id.clone(),
            source.trial_id.clone(),
            source.total_reward,
            source.env_total_reward,
            source.completed_waypoints.clone(),
            source.trajectory.clone(),
            source.decision_turns.clone(),
            source.inference_url.clone(),
            source.inference_error_count,
            source.last_inference_error.clone(),
        )
    };
    let save_data: SaveData = serde_json::from_slice(&checkpoint_bytes).map_err(|_| {
        AppError::new(
            StatusCode::BAD_REQUEST,
            "invalid_checkpoint_payload",
            "checkpoint bytes could not be parsed as crafter-rs SaveData",
        )
    })?;
    let mut env_config = source_env_config;
    env_config.extend(payload.overrides.env);
    env_config.extend(payload.overrides.env_config);
    let mut policy_config = source_policy_config;
    policy_config.extend(payload.overrides.policy);
    policy_config.extend(payload.overrides.policy_config);
    let target_rollout_id = payload
        .target_rollout_id
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("{}_resume_{}", rollout_id, &Uuid::new_v4().to_string()[..8]));
    let checkpoint_step = save_data.step;
    let mut target = RolloutRecord {
        rollout_id: target_rollout_id.clone(),
        trace_correlation_id: target_rollout_id.clone(),
        trial_id: source_trial_id,
        seed,
        session: save_data.into_session(),
        env_config,
        policy_config,
        planner_mode,
        waypoints,
        created_at: now_iso(),
        started_at: None,
        completed_at: None,
        status: "pending".to_string(),
        parent_rollout_id: Some(parent_rollout_id),
        total_reward,
        last_reward: 0.0,
        env_total_reward,
        env_last_reward: 0.0,
        last_done: false,
        last_done_reason: None,
        last_newly_unlocked: Vec::new(),
        trajectory: source_trajectory
            .into_iter()
            .filter(|item| item.step_idx <= checkpoint_step)
            .collect(),
        completed_waypoints: source_completed_waypoints
            .into_iter()
            .filter(|value| value.get("step").and_then(Value::as_u64).unwrap_or(0) <= checkpoint_step)
            .collect(),
        current_waypoint_index: 0,
        planner_failure_code: None,
        last_status_detail: None,
        decision_turns: source_decision_turns
            .into_iter()
            .filter(|item| item.step_end <= checkpoint_step as u64)
            .collect(),
        checkpoints: BTreeMap::new(),
        terminated: false,
        inference_url: source_inference_url,
        inference_error_count: source_inference_error_count,
        last_inference_error: source_last_inference_error,
        llm_call_count: 0,
        llm_action_batches: Vec::new(),
    };
    target.current_waypoint_index = target.completed_waypoints.len();
    update_waypoint_progress(&mut target);
    run_rollout_segment(&mut target, segment_steps).await;
    let response = rollout_response(&target);
    let mut store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    store.rollouts.insert(target_rollout_id, target);
    Ok((StatusCode::ACCEPTED, Json(response)))
}

async fn checkpoint_save(
    State(state): State<AppState>,
    Json(body): Json<CheckpointBody>,
) -> AppResult<(StatusCode, Json<Value>)> {
    let rollout_id = body
        .rollout_id
        .clone()
        .ok_or_else(|| AppError::new(StatusCode::UNPROCESSABLE_ENTITY, "missing_rollout_id", "rollout_id is required"))?;
    create_checkpoint(State(state), Path(rollout_id), Json(Some(body))).await
}

async fn checkpoint_list(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> AppResult<Json<Value>> {
    let rollout_id = params
        .get("rollout_id")
        .cloned()
        .ok_or_else(|| AppError::new(StatusCode::UNPROCESSABLE_ENTITY, "missing_rollout_id", "rollout_id query param is required"))?;
    list_checkpoints(State(state), Path(rollout_id)).await
}

async fn checkpoint_load(
    State(state): State<AppState>,
    Json(body): Json<ResumeBody>,
) -> AppResult<Json<Value>> {
    let rollout_id = body.rollout_id.clone().ok_or_else(|| {
        AppError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "missing_rollout_id",
            "rollout_id is required",
        )
    })?;
    let (_, Json(resource)) =
        resume_rollout(State(state), Path(rollout_id.clone()), Json(Some(body))).await?;
    Ok(Json(json!({
        "ok": true,
        "rollout_id": resource.get("rollout_id").cloned().unwrap_or(Value::String(rollout_id)),
        "status": resource.get("status").cloned().unwrap_or(Value::String("unknown".to_string())),
    })))
}

async fn checkpoint_dump_alias(
    State(state): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<Option<CheckpointBody>>,
) -> AppResult<Json<Value>> {
    let (_, Json(checkpoint)) =
        create_checkpoint(State(state.clone()), Path(rollout_id.clone()), Json(body)).await?;
    let checkpoint_id = checkpoint
        .get("checkpoint_id")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "missing_checkpoint_id", "checkpoint creation returned no checkpoint_id"))?;
    let store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    let record = store
        .rollouts
        .get(&rollout_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
    let checkpoint_record = record
        .checkpoints
        .get(checkpoint_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_checkpoint", format!("checkpoint not found: {}", checkpoint_id)))?;
    Ok(Json(json!({
        "ok": true,
        "rollout_id": rollout_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_data_base64": BASE64_STANDARD.encode(&checkpoint_record.checkpoint_bytes),
        "metadata": checkpoint_record.metadata,
    })))
}

async fn checkpoint_restore_alias(
    State(state): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<Option<ResumeBody>>,
) -> AppResult<Json<Value>> {
    let payload = body.unwrap_or_default();
    if payload
        .checkpoint_data_base64
        .as_ref()
        .is_some_and(|value| !value.trim().is_empty())
    {
        let mut store = state
            .inner
            .lock()
            .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
        let record = store
            .rollouts
            .get_mut(&rollout_id)
            .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
        let bytes = BASE64_STANDARD
            .decode(payload.checkpoint_data_base64.as_deref().unwrap_or_default())
            .map_err(|_| AppError::new(StatusCode::BAD_REQUEST, "invalid_checkpoint_data", "checkpoint_data_base64 could not be decoded"))?;
        let save_data: SaveData = serde_json::from_slice(&bytes).map_err(|_| {
            AppError::new(
                StatusCode::BAD_REQUEST,
                "invalid_checkpoint_payload",
                "checkpoint bytes could not be parsed as crafter-rs SaveData",
            )
        })?;
        record.session = save_data.into_session();
        record.status = "completed".to_string();
        record.completed_at = Some(now_iso());
        return Ok(Json(json!({
            "ok": true,
            "rollout_id": rollout_id,
            "restored": true,
            "current_state": payload_from_record(record),
        })));
    }

    let (status, Json(resource)) = resume_rollout(State(state), Path(rollout_id.clone()), Json(Some(payload))).await?;
    Ok(Json(json!({
        "ok": status == StatusCode::ACCEPTED,
        "rollout_id": resource.get("rollout_id").cloned().unwrap_or(Value::String(rollout_id)),
        "status": resource.get("status").cloned().unwrap_or(Value::String("unknown".to_string())),
        "metadata": resource.get("metadata").cloned().unwrap_or(Value::Object(Map::new())),
    })))
}

async fn terminate_rollout(
    State(state): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<Option<TerminateBody>>,
) -> AppResult<Json<Value>> {
    let mut store = state
        .inner
        .lock()
        .map_err(|_| AppError::new(StatusCode::INTERNAL_SERVER_ERROR, "lock_failed", "store lock poisoned"))?;
    let record = store
        .rollouts
        .get_mut(&rollout_id)
        .ok_or_else(|| AppError::new(StatusCode::NOT_FOUND, "unknown_rollout", format!("rollout not found: {}", rollout_id)))?;
    record.terminated = true;
    record.status = "terminated".to_string();
    record.completed_at = Some(now_iso());
    record.last_status_detail = body
        .and_then(|payload| payload.reason)
        .or_else(|| Some("terminated_by_request".to_string()));
    Ok(Json(json!({
        "ok": true,
        "rollout_id": rollout_id,
        "status": record.status,
    })))
}

async fn legacy_terminate(
    State(state): State<AppState>,
    Json(body): Json<TerminateBody>,
) -> AppResult<Json<Value>> {
    let env_id = body
        .env_id
        .clone()
        .ok_or_else(|| AppError::new(StatusCode::UNPROCESSABLE_ENTITY, "missing_env_id", "env_id is required"))?;
    terminate_rollout(State(state), Path(env_id), Json(Some(body))).await
}

fn task_info_payload(seed: Option<u64>) -> Value {
    let mut task_metadata = json!({
        "supports_waypoint_planner_rollouts": true,
        "supports_restart_from_seed_checkpoint": true,
        "supports_run_from_seed_checkpoint_until": true,
        "supports_terminate": true,
        "checkpoint_routes": [
            "/rollouts/{rollout_id}/checkpoints",
            "/rollouts/{rollout_id}/resume",
            "/rollouts/{rollout_id}/checkpoint/dump",
            "/rollouts/{rollout_id}/checkpoint/restore"
        ],
        "planner_modes": ["direct", "waypoint_planned"],
        "action_names": classic_action_names(),
        "default_episode_max_steps": DEFAULT_EPISODE_MAX_STEPS
    });
    if let Some(seed) = seed {
        task_metadata
            .as_object_mut()
            .expect("task metadata object")
            .insert("seed".to_string(), json!(seed));
    }
    json!({
        "task": {
            "id": "crafter_rs_container",
            "name": "Crafter RS Container",
            "description": "Long-horizon Crafter container backed by crafter-rs with checkpoint and resume support.",
            "version": "v0"
        },
        "dataset": {
            "id": "crafter_long_horizon",
            "name": "Crafter Long Horizon",
            "version": "v1",
            "splits": ["train", "validation", "eval"],
            "default_split": "train"
        },
        "inference": {
            "model": DEFAULT_MODEL
        },
        "limits": {
            "max_turns": DEFAULT_EPISODE_MAX_STEPS
        },
        "task_metadata": task_metadata
    })
}

fn session_config_from_env(seed: u64, env_config: &Map<String, Value>) -> SessionConfig {
    let mut config = match env_config.get("difficulty").and_then(Value::as_str) {
        Some("easy") => SessionConfig::easy(),
        Some("hard") => SessionConfig::hard(),
        _ => SessionConfig::default(),
    };
    config.seed = Some(seed);
    config.max_steps = Some(as_u32(
        env_config.get("episode_max_steps").or_else(|| env_config.get("max_steps")),
        DEFAULT_EPISODE_MAX_STEPS,
    ));
    config.view_radius = as_u32(env_config.get("view_radius"), DEFAULT_VIEW_RADIUS).max(1);
    config.full_world_state = as_bool(env_config.get("full_world_state"), false);
    if as_bool(env_config.get("craftax_enabled"), false) {
        config.craftax.enabled = true;
    }
    config
}

fn derive_seed(trace_correlation_id: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    trace_correlation_id.hash(&mut hasher);
    hasher.finish()
}

fn segment_steps_from_config(env_config: &Map<String, Value>) -> u32 {
    as_u32(
        env_config
            .get("segment_steps")
            .or_else(|| env_config.get("run_until_step_delta"))
            .or_else(|| env_config.get("max_steps")),
        DEFAULT_EPISODE_MAX_STEPS,
    )
    .max(1)
}

fn now_iso() -> String {
    Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn classic_action_names() -> Vec<&'static str> {
    vec![
        "noop",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "do",
        "sleep",
        "place_stone",
        "place_table",
        "place_furnace",
        "place_plant",
        "make_wood_pickaxe",
        "make_stone_pickaxe",
        "make_iron_pickaxe",
        "make_wood_sword",
        "make_stone_sword",
        "make_iron_sword",
    ]
}

fn action_name(action: Action) -> &'static str {
    match action {
        Action::Noop => "noop",
        Action::MoveLeft => "move_left",
        Action::MoveRight => "move_right",
        Action::MoveUp => "move_up",
        Action::MoveDown => "move_down",
        Action::Do => "do",
        Action::Sleep => "sleep",
        Action::PlaceStone => "place_stone",
        Action::PlaceTable => "place_table",
        Action::PlaceFurnace => "place_furnace",
        Action::PlacePlant => "place_plant",
        Action::MakeWoodPickaxe => "make_wood_pickaxe",
        Action::MakeStonePickaxe => "make_stone_pickaxe",
        Action::MakeIronPickaxe => "make_iron_pickaxe",
        Action::MakeWoodSword => "make_wood_sword",
        Action::MakeStoneSword => "make_stone_sword",
        Action::MakeIronSword => "make_iron_sword",
        Action::MakeDiamondPickaxe => "make_diamond_pickaxe",
        Action::MakeDiamondSword => "make_diamond_sword",
        Action::MakeIronArmor => "make_iron_armor",
        Action::MakeDiamondArmor => "make_diamond_armor",
        Action::MakeBow => "make_bow",
        Action::MakeArrow => "make_arrow",
        Action::ShootArrow => "shoot_arrow",
        Action::DrinkPotionRed => "drink_potion_red",
        Action::DrinkPotionGreen => "drink_potion_green",
        Action::DrinkPotionBlue => "drink_potion_blue",
        Action::DrinkPotionPink => "drink_potion_pink",
        Action::DrinkPotionCyan => "drink_potion_cyan",
        Action::DrinkPotionYellow => "drink_potion_yellow",
    }
}

fn action_from_value(value: &Value) -> Option<Action> {
    match value {
        Value::Number(number) => number
            .as_u64()
            .and_then(|idx| Action::from_index(idx as u8)),
        Value::String(name) => classic_actions()
            .into_iter()
            .find(|action| action_name(*action) == name.trim().to_lowercase()),
        _ => None,
    }
}

fn classic_actions() -> Vec<Action> {
    Action::classic_actions().to_vec()
}

struct InferenceResponseData {
    content: String,
    reasoning_text: Option<String>,
    request_id: Option<String>,
    sequence_logprob: Option<f64>,
    usage: Value,
}

struct ActionBatchDecision {
    actions: Vec<Action>,
    prompt_messages: Vec<Value>,
    assistant_text: String,
    reasoning_text: Option<String>,
    trainable: bool,
    invalid_parse: bool,
    request_id: Option<String>,
    sequence_logprob: Option<f64>,
    usage: Value,
}

async fn next_action_batch(record: &mut RolloutRecord) -> Result<ActionBatchDecision, String> {
    let policy_config = &record.policy_config;
    let step_count = record.session.get_state().step;
    if let Some(Value::Array(items)) = policy_config.get("actions") {
        let actions: Vec<Action> = items.iter().filter_map(action_from_value).collect();
        if !actions.is_empty() {
            let action_names = actions.iter().map(|action| action_name(*action).to_string()).collect::<Vec<_>>();
            return Ok(ActionBatchDecision {
                actions,
                prompt_messages: Vec::new(),
                assistant_text: json!({ "actions": action_names }).to_string(),
                reasoning_text: None,
                trainable: false,
                invalid_parse: false,
                request_id: None,
                sequence_logprob: None,
                usage: json!({}),
            });
        }
    }
    if let Some(Value::Array(items)) = policy_config.get("action_cycle") {
        let cycle: Vec<Action> = items.iter().filter_map(action_from_value).collect();
        if !cycle.is_empty() {
            let action = cycle[(step_count as usize) % cycle.len()];
            return Ok(ActionBatchDecision {
                actions: vec![action],
                prompt_messages: Vec::new(),
                assistant_text: json!({ "actions": [action_name(action)] }).to_string(),
                reasoning_text: None,
                trainable: false,
                invalid_parse: false,
                request_id: None,
                sequence_logprob: None,
                usage: json!({}),
            });
        }
    }
    if policy_config.get("model").is_some() || policy_config.get("inference_url").is_some() {
        match infer_actions(record).await {
            Ok(batch) if !batch.actions.is_empty() => {
                record.llm_call_count += 1;
                record
                    .llm_action_batches
                    .push(batch.actions.iter().map(|action| action_name(*action).to_string()).collect());
                if batch.invalid_parse {
                    record.inference_error_count += 1;
                    record.last_inference_error = Some("model output required repair".to_string());
                } else {
                    record.last_inference_error = None;
                }
                return Ok(batch);
            }
            Ok(_) => {
                record.inference_error_count += 1;
                let error = "inference returned an empty action batch".to_string();
                record.last_inference_error = Some(error.clone());
                return Err(error);
            }
            Err(error) => {
                record.inference_error_count += 1;
                record.last_inference_error = Some(error.clone());
                return Err(error);
            }
        }
    }
    Err("policy config missing actions, action_cycle, model, or inference_url".to_string())
}

async fn infer_actions(record: &mut RolloutRecord) -> Result<ActionBatchDecision, String> {
    let inference_url = record
        .policy_config
        .get("inference_url")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(DEFAULT_OPENROUTER_URL)
        .to_string();
    let model = record
        .policy_config
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(DEFAULT_MODEL)
        .to_string();
    let api_key = record
        .policy_config
        .get("api_key")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .ok_or_else(|| "missing OPENROUTER_API_KEY/OPENAI_API_KEY for inference".to_string())?;
    let system_prompt = record
        .policy_config
        .get("system_prompt")
        .or_else(|| record.policy_config.get("instruction"))
        .or_else(|| record.policy_config.get("prompt"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(DEFAULT_POLICY_PROMPT)
        .to_string();
    let timeout_s = as_u64(record.policy_config.get("timeout_s"), 45).max(1);
    let max_tokens = as_u64(record.policy_config.get("max_tokens"), 180).max(32);
    let temperature = as_f64(record.policy_config.get("temperature"), 0.0);
    let enable_thinking = as_bool(record.policy_config.get("enable_thinking"), false);
    let target_batch_size = as_u64(record.policy_config.get("target_action_batch_size"), 4)
        .clamp(2, 6) as usize;
    let min_batch_size = as_u64(record.policy_config.get("min_action_batch_size"), 3)
        .clamp(2, target_batch_size as u64) as usize;
    let available_actions = available_actions_for_state(&record.session.config);
    let observation_text = build_observation_text(record);

    let user_prompt = format!(
        "Current Crafter long-horizon observation:\n{}\n\nPlan a short useful macro-action. Return exactly {} actions unless the environment is already done. Avoid one-action plans. Use only these actions: {}.\nReturn strict JSON only, for example {{\"actions\":[\"move_right\",\"move_right\",\"move_down\",\"do\"]}}.",
        observation_text,
        target_batch_size,
        available_actions.join(", ")
    );
    let initial_messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": user_prompt}),
    ];

    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_s))
        .build()
        .map_err(|error| format!("failed to build reqwest client: {}", error))?;
    let first_response = send_inference_request(
        &client,
        &inference_url,
        &api_key,
        &model,
        temperature,
        max_tokens,
        initial_messages.clone(),
        Some(enable_thinking),
    )
    .await?;

    record.inference_url = Some(inference_url.clone());
    let mut final_messages = initial_messages;
    let mut final_content = first_response.content.clone();
    let mut final_reasoning_text = first_response.reasoning_text.clone();
    let mut final_request_id = first_response.request_id.clone();
    let mut final_sequence_logprob = first_response.sequence_logprob;
    let mut final_usage = first_response.usage.clone();
    let mut invalid_parse = false;
    let mut actions = parse_actions_from_text(&first_response.content);
    if actions.len() < min_batch_size {
        let repair_prompt = format!(
            "Your previous response was invalid because it contained {} action(s), but I need at least {} and preferably exactly {}.\nPrevious response:\n{}\n\nRewrite it as strict JSON with exactly {} valid actions chosen from: {}.\nDo not explain anything.",
            actions.len(),
            min_batch_size,
            target_batch_size,
            first_response.content,
            target_batch_size,
            available_actions.join(", ")
        );
        let repaired_messages = vec![
            json!({"role": "system", "content": system_prompt}),
            json!({"role": "user", "content": user_prompt}),
            json!({"role": "assistant", "content": first_response.content}),
            json!({"role": "user", "content": repair_prompt}),
        ];
        let repaired_response = send_inference_request(
            &client,
            &inference_url,
            &api_key,
            &model,
            temperature,
            max_tokens,
            repaired_messages.clone(),
            Some(enable_thinking),
        )
        .await?;
        final_messages = repaired_messages;
        final_content = repaired_response.content.clone();
        final_reasoning_text = repaired_response.reasoning_text.clone();
        final_request_id = repaired_response.request_id.clone();
        final_sequence_logprob = repaired_response.sequence_logprob;
        final_usage = repaired_response.usage.clone();
        actions = parse_actions_from_text(&repaired_response.content);
        invalid_parse = true;
    }

    record.last_inference_error = None;
    if actions.is_empty() {
        return Err(format!(
            "model output could not be parsed into any valid Crafter actions: {}",
            final_content
        ));
    }
    if actions.len() < min_batch_size {
        return Err(format!(
            "model output contained {} valid Crafter action(s), fewer than required minimum {}: {}",
            actions.len(),
            min_batch_size,
            final_content
        ));
    }
    Ok(ActionBatchDecision {
        actions,
        prompt_messages: final_messages,
        assistant_text: final_content,
        reasoning_text: final_reasoning_text,
        trainable: !invalid_parse,
        invalid_parse,
        request_id: final_request_id,
        sequence_logprob: final_sequence_logprob,
        usage: final_usage,
    })
}

async fn send_inference_request(
    client: &Client,
    inference_url: &str,
    api_key: &str,
    model: &str,
    temperature: f64,
    max_tokens: u64,
    messages: Vec<Value>,
    enable_thinking: Option<bool>,
) -> Result<InferenceResponseData, String> {
    let mut request_variants = Vec::new();
    match enable_thinking {
        Some(flag) => {
            request_variants.push((true, Some(flag)));
            request_variants.push((false, Some(flag)));
            request_variants.push((true, None));
            request_variants.push((false, None));
        }
        None => {
            request_variants.push((true, None));
            request_variants.push((false, None));
        }
    }
    for (request_logprobs, request_enable_thinking) in request_variants {
        let mut request_body = json!({
            "model": model,
            "messages": messages.clone(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": { "type": "json_object" }
        });
        if request_logprobs {
            request_body["logprobs"] = Value::Bool(true);
        }
        if let Some(flag) = request_enable_thinking {
            request_body["chat_template_kwargs"] = json!({
                "enable_thinking": flag
            });
        }
        let response = client
            .post(inference_url)
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, format!("Bearer {}", api_key))
            .header("HTTP-Referer", "https://github.com/synth-laboratories")
            .header("X-Title", "crafter-rs-container")
            .json(&request_body)
            .send()
            .await
            .map_err(|error| format!("inference request failed: {}", error))?;

        let status = response.status();
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("failed to parse inference response json: {}", error))?;
        if !status.is_success() {
            let retry_without_logprobs = request_logprobs && matches!(status.as_u16(), 400 | 403 | 404 | 422 | 501);
            if retry_without_logprobs {
                continue;
            }
            return Err(format!("inference status {} body {}", status, truncate_json(&body)));
        }
        return Ok(InferenceResponseData {
            content: extract_model_content(&body),
            reasoning_text: extract_model_reasoning(&body),
            request_id: request_id.or_else(|| body.get("id").and_then(Value::as_str).map(str::to_string)),
            sequence_logprob: extract_model_sequence_logprob(&body),
            usage: body.get("usage").cloned().unwrap_or_else(|| json!({})),
        });
    }
    Err("inference provider rejected all supported request variants".to_string())
}

fn extract_model_sequence_logprob(body: &Value) -> Option<f64> {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())?;

    let direct_candidates = [
        choice.get("sequence_logprob"),
        choice.get("assistant_sequence_logprob"),
        choice.get("old_logprob"),
    ];
    for candidate in direct_candidates {
        if let Some(value) = candidate.and_then(Value::as_f64) {
            return Some(value);
        }
    }

    let array_candidates = [
        choice.get("logprobs").and_then(|value| value.get("content")),
        choice.get("logprobs").and_then(|value| value.get("token_logprobs")),
        choice
            .get("message")
            .and_then(|value| value.get("logprobs"))
            .and_then(|value| value.get("content")),
    ];
    for candidate in array_candidates {
        if let Some(items) = candidate.and_then(Value::as_array) {
            if let Some(total) = sum_logprob_items(items) {
                return Some(total);
            }
        }
    }

    None
}

fn sum_logprob_items(items: &[Value]) -> Option<f64> {
    let mut total = 0.0f64;
    let mut seen = false;
    for item in items {
        if let Some(value) = item.as_f64() {
            total += value;
            seen = true;
            continue;
        }
        if let Some(value) = item.get("logprob").and_then(Value::as_f64) {
            total += value;
            seen = true;
            continue;
        }
    }
    if seen {
        Some(total)
    } else {
        None
    }
}

fn build_observation_text(record: &RolloutRecord) -> String {
    let state = record.session.get_state();
    let inventory = inventory_json(&state.inventory);
    let achievements = unlocked_achievements(record);
    let available_actions = available_actions_for_state(&record.session.config);
    let hints = hints_for_state(&state);
    let map_legend = map_legend_for_state(&record.session.config);
    let recent_actions = recent_action_history(record, 8);
    let waypoint = record.waypoints.get(record.current_waypoint_index);
    let ascii_view = TextRenderer::minimal()
        .render(&state)
        .unwrap_or_else(|_| String::new());
    let mut lines = vec![
        format!("step={}", state.step),
        format!("daylight={:.4}", state.daylight),
        format!("player_pos={:?}", state.player_pos),
        format!("player_facing={:?}", state.player_facing),
        format!("sleeping={}", state.player_sleeping),
        format!("inventory={}", inventory),
        format!("achievements={}", serde_json::to_string(&achievements).unwrap_or_else(|_| "[]".to_string())),
        format!("available_actions={}", serde_json::to_string(&available_actions).unwrap_or_else(|_| "[]".to_string())),
        format!("hints={}", serde_json::to_string(&hints).unwrap_or_else(|_| "[]".to_string())),
        format!("map_legend={}", serde_json::to_string(&map_legend).unwrap_or_else(|_| "{}".to_string())),
    ];
    if !ascii_view.trim().is_empty() {
        lines.push("ascii_view:".to_string());
        lines.push(ascii_view);
    }
    if !recent_actions.is_empty() {
        lines.push("recent_action_history:".to_string());
        lines.extend(recent_actions);
    }
    if record.planner_mode.as_deref() == Some("waypoint_planned") {
        lines.push(format!("completed_waypoints={}", record.completed_waypoints.len()));
        if let Some(waypoint) = waypoint {
            lines.push(format!("current_waypoint={}", waypoint.description));
            if let Some(achievement) = waypoint.achievement.as_ref() {
                lines.push(format!("current_waypoint_achievement={}", achievement));
            }
            if !waypoint.inventory_requirements.is_empty() {
                lines.push(format!(
                    "current_waypoint_inventory={}",
                    serde_json::to_string(&waypoint.inventory_requirements).unwrap_or_else(|_| "{}".to_string())
                ));
            }
            if let Some(position) = waypoint.player_position {
                lines.push(format!("current_waypoint_position={:?}", position));
            }
        }
    }
    lines.join("\n")
}

fn available_actions_for_state(config: &SessionConfig) -> Vec<String> {
    let mut available_actions = vec![
        "move_left".to_string(),
        "move_right".to_string(),
        "move_up".to_string(),
        "move_down".to_string(),
        "do".to_string(),
        "sleep".to_string(),
        "place_table".to_string(),
        "place_stone".to_string(),
        "place_furnace".to_string(),
        "place_plant".to_string(),
        "make_wood_pickaxe".to_string(),
        "make_stone_pickaxe".to_string(),
        "make_iron_pickaxe".to_string(),
        "make_wood_sword".to_string(),
        "make_stone_sword".to_string(),
        "make_iron_sword".to_string(),
    ];
    if config.craftax.enabled && config.craftax.items_enabled {
        available_actions.extend([
            "make_diamond_pickaxe".to_string(),
            "make_diamond_sword".to_string(),
            "make_iron_armor".to_string(),
            "make_diamond_armor".to_string(),
            "make_bow".to_string(),
            "make_arrow".to_string(),
            "shoot_arrow".to_string(),
        ]);
        if config.craftax.potions_enabled {
            available_actions.extend([
                "drink_potion_red".to_string(),
                "drink_potion_green".to_string(),
                "drink_potion_blue".to_string(),
                "drink_potion_pink".to_string(),
                "drink_potion_cyan".to_string(),
                "drink_potion_yellow".to_string(),
            ]);
        }
    }
    available_actions
}

fn hints_for_state(state: &crafter_core::GameState) -> Vec<String> {
    let inv = &state.inventory;
    let ach = &state.achievements;
    let mut hints = Vec::new();
    if inv.wood < 2 && inv.wood_pickaxe == 0 {
        hints.push("Collect wood by moving next to a tree and using 'do'.".to_string());
    }
    if inv.wood >= 2 && ach.place_table == 0 {
        hints.push("Place a crafting table with 'place_table' after gathering enough wood.".to_string());
    }
    if inv.wood >= 1 && inv.wood_pickaxe == 0 && ach.place_table > 0 {
        hints.push("Make a wood pickaxe with 'make_wood_pickaxe' while near the table.".to_string());
    }
    if inv.wood >= 1 && inv.wood_sword == 0 && ach.place_table > 0 {
        hints.push("Make a wood sword with 'make_wood_sword' while near the table.".to_string());
    }
    if inv.health < 5 && inv.food > 2 {
        hints.push("Use 'sleep' to restore health when safe.".to_string());
    }
    hints
}

fn map_legend_for_state(config: &SessionConfig) -> BTreeMap<String, String> {
    let mut map_legend = BTreeMap::from([
        ("@".to_string(), "Player".to_string()),
        (".".to_string(), "Grass".to_string()),
        ("~".to_string(), "Water".to_string()),
        ("#".to_string(), "Stone".to_string()),
        ("_".to_string(), "Path".to_string()),
        (":".to_string(), "Sand".to_string()),
        ("T".to_string(), "Tree".to_string()),
        ("%".to_string(), "Lava".to_string()),
        ("c".to_string(), "Coal".to_string()),
        ("i".to_string(), "Iron".to_string()),
        ("D".to_string(), "Diamond".to_string()),
        ("+".to_string(), "Table".to_string()),
        ("F".to_string(), "Furnace".to_string()),
        ("C".to_string(), "Cow".to_string()),
        ("Z".to_string(), "Zombie".to_string()),
        ("S".to_string(), "Skeleton".to_string()),
        ("p".to_string(), "Plant".to_string()),
        ("P".to_string(), "Ripe Plant".to_string()),
        ("*".to_string(), "Projectile".to_string()),
    ]);
    if config.craftax.enabled {
        map_legend.extend([
            ("s".to_string(), "Sapphire".to_string()),
            ("r".to_string(), "Ruby".to_string()),
            ("H".to_string(), "Chest".to_string()),
            ("O".to_string(), "Orc".to_string()),
            ("M".to_string(), "Orc Mage".to_string()),
            ("K".to_string(), "Knight".to_string()),
            ("A".to_string(), "Knight Archer".to_string()),
            ("t".to_string(), "Troll".to_string()),
            ("B".to_string(), "Bat".to_string()),
            ("N".to_string(), "Snail".to_string()),
        ]);
    }
    map_legend
}

fn recent_action_history(record: &RolloutRecord, n: usize) -> Vec<String> {
    record
        .trajectory
        .iter()
        .rev()
        .take(n)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|item| {
            format!(
                "step={} actions={:?} reward={} total_reward={} pos={:?} achievements={:?}",
                item.step_idx, item.actions, item.reward, item.total_reward, item.player_pos, item.achievements
            )
        })
        .collect()
}

fn inventory_json(inventory: &crafter_core::Inventory) -> String {
    serde_json::to_string(&inventory_payload(inventory))
        .unwrap_or_else(|_| "{}".to_string())
}

fn inventory_payload(inventory: &crafter_core::Inventory) -> Value {
    json!({
        "health": inventory.health,
        "food": inventory.food,
        "drink": inventory.drink,
        "energy": inventory.energy,
        "sapling": inventory.sapling,
        "wood": inventory.wood,
        "stone": inventory.stone,
        "coal": inventory.coal,
        "iron": inventory.iron,
        "diamond": inventory.diamond,
        "sapphire": inventory.sapphire,
        "ruby": inventory.ruby,
        "wood_pickaxe": inventory.wood_pickaxe,
        "stone_pickaxe": inventory.stone_pickaxe,
        "iron_pickaxe": inventory.iron_pickaxe,
        "diamond_pickaxe": inventory.diamond_pickaxe,
        "wood_sword": inventory.wood_sword,
        "stone_sword": inventory.stone_sword,
        "iron_sword": inventory.iron_sword,
        "diamond_sword": inventory.diamond_sword,
        "bow": inventory.bow,
        "arrows": inventory.arrows,
        "xp": inventory.xp,
        "level": inventory.level,
        "stat_points": inventory.stat_points,
    })
}

fn extract_model_content(body: &Value) -> String {
    if let Some(content) = extract_message_field(body, "content") {
        match content {
            Value::String(text) => return text.clone(),
            Value::Array(parts) => {
                let text = parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    return text;
                }
            }
            _ => {}
        }
    }
    if let Some(reasoning) = extract_model_reasoning(body) {
        if !reasoning.is_empty() {
            return reasoning;
        }
    }
    body.to_string()
}

fn extract_model_reasoning(body: &Value) -> Option<String> {
    let reasoning = extract_message_field(body, "reasoning")?;
    match reasoning {
        Value::String(text) => {
            if text.trim().is_empty() {
                None
            } else {
                Some(text.clone())
            }
        }
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        _ => None,
    }
}

fn extract_message_field<'a>(body: &'a Value, field_name: &str) -> Option<&'a Value> {
    body.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get(field_name))
}

fn parse_actions_from_text(content: &str) -> Vec<Action> {
    if content.trim().is_empty() {
        return Vec::new();
    }
    if let Ok(parsed) = serde_json::from_str::<Value>(content) {
        let actions = parse_actions_from_value(&parsed);
        if !actions.is_empty() {
            return actions;
        }
    }
    if let (Some(start), Some(end)) = (content.find('{'), content.rfind('}')) {
        if start < end {
            if let Ok(parsed) = serde_json::from_str::<Value>(&content[start..=end]) {
                let actions = parse_actions_from_value(&parsed);
                if !actions.is_empty() {
                    return actions;
                }
            }
        }
    }
    let lowered = content.to_lowercase();
    classic_actions()
        .into_iter()
        .filter(|action| lowered.contains(action_name(*action)))
        .take(6)
        .collect()
}

fn parse_actions_from_value(value: &Value) -> Vec<Action> {
    value
        .get("actions")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(action_from_value).take(6).collect())
        .unwrap_or_default()
}

fn truncate_json(body: &Value) -> String {
    let text = body.to_string();
    if text.len() > 300 {
        format!("{}...", &text[..300])
    } else {
        text
    }
}

fn parse_waypoints(raw_request: Option<&Value>) -> Vec<Waypoint> {
    let Some(request) = raw_request else {
        return Vec::new();
    };
    let raw_items = request
        .get("waypoints")
        .or_else(|| request.get("subgoals"))
        .or_else(|| request.get("plan").and_then(|plan| plan.get("steps")));
    let Some(Value::Array(items)) = raw_items else {
        return Vec::new();
    };

    items
        .iter()
        .filter_map(|item| match item {
            Value::String(description) if !description.trim().is_empty() => Some(Waypoint {
                description: description.trim().to_string(),
                achievement: None,
                inventory_requirements: BTreeMap::new(),
                player_position: None,
            }),
            Value::Object(map) => {
                let description = map
                    .get("description")
                    .or_else(|| map.get("label"))
                    .or_else(|| map.get("goal"))
                    .or_else(|| map.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if description.is_empty() {
                    return None;
                }
                let inventory_requirements = map
                    .get("inventory")
                    .and_then(Value::as_object)
                    .map(|inventory| {
                        inventory
                            .iter()
                            .map(|(key, value)| (key.clone(), as_i64(Some(value), 0).max(0)))
                            .collect::<BTreeMap<_, _>>()
                    })
                    .unwrap_or_default();
                let player_position = map
                    .get("player_position")
                    .or_else(|| map.get("position"))
                    .and_then(Value::as_array)
                    .and_then(|values| {
                        if values.len() < 2 {
                            return None;
                        }
                        Some((
                            as_i32(values.first(), 0),
                            as_i32(values.get(1), 0),
                        ))
                    });
                Some(Waypoint {
                    description,
                    achievement: map
                        .get("achievement")
                        .and_then(Value::as_str)
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty()),
                    inventory_requirements,
                    player_position,
                })
            }
            _ => None,
        })
        .collect()
}

fn unlocked_achievements(record: &RolloutRecord) -> Vec<String> {
    let state = record.session.get_state();
    let achievements = state.achievements;
    let pairs = [
        ("collect_coal", achievements.collect_coal),
        ("collect_diamond", achievements.collect_diamond),
        ("collect_drink", achievements.collect_drink),
        ("collect_iron", achievements.collect_iron),
        ("collect_sapling", achievements.collect_sapling),
        ("collect_stone", achievements.collect_stone),
        ("collect_wood", achievements.collect_wood),
        ("defeat_skeleton", achievements.defeat_skeleton),
        ("defeat_zombie", achievements.defeat_zombie),
        ("eat_cow", achievements.eat_cow),
        ("eat_plant", achievements.eat_plant),
        ("make_iron_pickaxe", achievements.make_iron_pickaxe),
        ("make_iron_sword", achievements.make_iron_sword),
        ("make_stone_pickaxe", achievements.make_stone_pickaxe),
        ("make_stone_sword", achievements.make_stone_sword),
        ("make_wood_pickaxe", achievements.make_wood_pickaxe),
        ("make_wood_sword", achievements.make_wood_sword),
        ("place_furnace", achievements.place_furnace),
        ("place_plant", achievements.place_plant),
        ("place_stone", achievements.place_stone),
        ("place_table", achievements.place_table),
        ("wake_up", achievements.wake_up),
        ("collect_sapphire", achievements.collect_sapphire),
        ("collect_ruby", achievements.collect_ruby),
        ("open_chest", achievements.open_chest),
        ("make_diamond_pickaxe", achievements.make_diamond_pickaxe),
        ("make_diamond_sword", achievements.make_diamond_sword),
        ("make_bow", achievements.make_bow),
        ("make_arrow", achievements.make_arrow),
        ("make_iron_armor", achievements.make_iron_armor),
        ("make_diamond_armor", achievements.make_diamond_armor),
        ("defeat_orc_soldier", achievements.defeat_orc_soldier),
        ("defeat_orc_mage", achievements.defeat_orc_mage),
        ("defeat_knight", achievements.defeat_knight),
        ("defeat_knight_archer", achievements.defeat_knight_archer),
        ("defeat_troll", achievements.defeat_troll),
        ("drink_potion", achievements.drink_potion),
        ("gain_xp", achievements.gain_xp),
        ("reach_level", achievements.reach_level),
    ];
    pairs
        .into_iter()
        .filter_map(|(name, count)| (count > 0).then(|| name.to_string()))
        .collect()
}

fn unique_achievement_reward(items: &[String]) -> f64 {
    let unique = items
        .iter()
        .map(|item| item.trim())
        .filter(|item| !item.is_empty())
        .collect::<HashSet<_>>();
    unique.len() as f64
}

fn cumulative_achievement_reward(record: &RolloutRecord) -> f64 {
    unlocked_achievements(record).len() as f64
}

fn update_waypoint_progress(record: &mut RolloutRecord) {
    loop {
        let Some(waypoint) = record.waypoints.get(record.current_waypoint_index).cloned() else {
            return;
        };
        let state = record.session.get_state();
        let unlocked = unlocked_achievements(record);
        let inventory = state.inventory;
        let inventory_completed = waypoint.inventory_requirements.iter().all(|(key, expected)| {
            inventory_value(&inventory, key) >= *expected
        });
        let position_completed = waypoint
            .player_position
            .map(|target| target == state.player_pos)
            .unwrap_or(false);
        let achievement_completed = waypoint
            .achievement
            .as_ref()
            .map(|name| unlocked.iter().any(|value| value == name))
            .unwrap_or(false);

        let reason = if achievement_completed {
            waypoint
                .achievement
                .clone()
                .map(|name| format!("achievement:{}", name))
        } else if !waypoint.inventory_requirements.is_empty() && inventory_completed {
            Some("inventory".to_string())
        } else if waypoint.player_position.is_some() && position_completed {
            Some("position".to_string())
        } else {
            None
        };

        let Some(reason) = reason else {
            return;
        };
        record.completed_waypoints.push(json!({
            "index": record.current_waypoint_index,
            "description": waypoint.description,
            "reason": reason,
            "step": record.session.get_state().step,
        }));
        record.current_waypoint_index += 1;
    }
}

fn inventory_value(inventory: &crafter_core::Inventory, name: &str) -> i64 {
    match name {
        "health" => inventory.health as i64,
        "food" => inventory.food as i64,
        "drink" => inventory.drink as i64,
        "energy" => inventory.energy as i64,
        "sapling" => inventory.sapling as i64,
        "wood" => inventory.wood as i64,
        "stone" => inventory.stone as i64,
        "coal" => inventory.coal as i64,
        "iron" => inventory.iron as i64,
        "diamond" => inventory.diamond as i64,
        "sapphire" => inventory.sapphire as i64,
        "ruby" => inventory.ruby as i64,
        "wood_pickaxe" => inventory.wood_pickaxe as i64,
        "stone_pickaxe" => inventory.stone_pickaxe as i64,
        "iron_pickaxe" => inventory.iron_pickaxe as i64,
        "diamond_pickaxe" => inventory.diamond_pickaxe as i64,
        "wood_sword" => inventory.wood_sword as i64,
        "stone_sword" => inventory.stone_sword as i64,
        "iron_sword" => inventory.iron_sword as i64,
        "diamond_sword" => inventory.diamond_sword as i64,
        "bow" => inventory.bow as i64,
        "arrows" => inventory.arrows as i64,
        "xp" => inventory.xp as i64,
        "level" => inventory.level as i64,
        "stat_points" => inventory.stat_points as i64,
        _ => 0,
    }
}

async fn run_rollout_segment(record: &mut RolloutRecord, segment_steps: u32) {
    record.status = "running".to_string();
    record.started_at.get_or_insert_with(now_iso);
    let mut steps_executed = 0u32;

    while steps_executed < segment_steps && !record.last_done && !record.terminated {
        let decision = match next_action_batch(record).await {
            Ok(decision) => decision,
            Err(error) => {
                record.status = "failed".to_string();
                record.last_status_detail = Some(format!("inference_error: {}", error));
                record.completed_at = Some(now_iso());
                return;
            }
        };
        let reward_before = record.total_reward;
        let env_reward_before = record.env_total_reward;
        let step_start = record.session.get_state().step;
        for action in &decision.actions {
            if steps_executed >= segment_steps || record.terminated {
                break;
            }
            let result = record.session.step(*action);
            apply_step_result(record, *action, &result);
            steps_executed += 1;
            if result.done {
                break;
            }
        }
        let reward_after = record.total_reward;
        let env_reward_after = record.env_total_reward;
        let step_end = record.session.get_state().step;
        record.decision_turns.push(DecisionTurnRecord {
            turn_index: record.decision_turns.len(),
            prompt_messages: decision.prompt_messages,
            assistant_text: decision.assistant_text,
            reasoning_text: decision.reasoning_text,
            actions: decision
                .actions
                .iter()
                .map(|action| action_name(*action).to_string())
                .collect(),
            decision_reward: reward_after - reward_before,
            reward_before,
            reward_after,
            env_reward_before,
            env_reward_after,
            step_start,
            step_end,
            trainable: decision.trainable,
            invalid_parse: decision.invalid_parse,
            behavior_version: policy_behavior_version(record),
            behavior_model: policy_model(record),
            route: policy_route(record),
            request_id: decision.request_id,
            behavior_sequence_logprob: decision.sequence_logprob,
            usage: decision.usage,
        });
    }

    if record.terminated {
        record.status = "terminated".to_string();
        record.last_status_detail = Some("terminated_by_request".to_string());
        record.completed_at = Some(now_iso());
        return;
    }
    if record.status == "failed" {
        if record.completed_at.is_none() {
            record.completed_at = Some(now_iso());
        }
        return;
    }

    record.status = "completed".to_string();
    record.completed_at = Some(now_iso());
    if record.last_done {
        record.last_status_detail = Some(
            record
                .last_done_reason
                .clone()
                .unwrap_or_else(|| "episode_done".to_string()),
        );
    } else if steps_executed >= segment_steps {
        record.last_status_detail = Some("segment_limit_reached".to_string());
    } else {
        record.last_status_detail = Some("ready".to_string());
    }

    if record.planner_mode.as_deref() == Some("waypoint_planned")
        && !record.waypoints.is_empty()
        && record.current_waypoint_index < record.waypoints.len()
        && (record.last_done || steps_executed >= segment_steps)
    {
        record.planner_failure_code = Some("waypoint_unreachable".to_string());
    } else {
        record.planner_failure_code = None;
    }
}

fn apply_step_result(record: &mut RolloutRecord, action: Action, result: &StepResult) {
    record.env_last_reward = result.reward as f64;
    record.env_total_reward += result.reward as f64;
    record.last_reward = unique_achievement_reward(&result.newly_unlocked);
    record.total_reward = cumulative_achievement_reward(record);
    record.last_done = result.done;
    record.last_done_reason = result.done_reason.as_ref().map(done_reason_name);
    record.last_newly_unlocked = result.newly_unlocked.clone();
    update_waypoint_progress(record);
    record.trajectory.push(TrajectoryItem {
        step_idx: result.state.step,
        actions: vec![action_name(action).to_string()],
        reward: record.last_reward,
        total_reward: record.total_reward,
        env_reward: record.env_last_reward,
        env_total_reward: record.env_total_reward,
        done: result.done,
        player_pos: result.state.player_pos,
        achievements: unlocked_achievements(record),
        newly_unlocked: result.newly_unlocked.clone(),
        inventory: inventory_payload(&result.state.inventory),
        current_waypoint_index: record.current_waypoint_index,
    });
}

fn done_reason_name(reason: &crafter_core::session::DoneReason) -> String {
    match reason {
        crafter_core::session::DoneReason::Death => "death".to_string(),
        crafter_core::session::DoneReason::MaxSteps => "max_steps".to_string(),
        crafter_core::session::DoneReason::Reset => "reset".to_string(),
    }
}

fn payload_from_record(record: &RolloutRecord) -> Value {
    let state = record.session.get_state();
    let ascii_view = TextRenderer::minimal()
        .render(&state)
        .unwrap_or_else(|_| String::new());
    json!({
        "state": state,
        "ascii_view": ascii_view,
        "reward": record.last_reward,
        "total_reward": record.total_reward,
        "env_reward": record.env_last_reward,
        "env_total_reward": record.env_total_reward,
        "done": record.last_done,
        "done_reason": record.last_done_reason,
        "newly_unlocked": record.last_newly_unlocked,
        "achievements": unlocked_achievements(record),
        "inventory": inventory_payload(&state.inventory),
        "player_pos": state.player_pos,
        "available_actions": classic_action_names(),
    })
}

fn make_checkpoint(record: &mut RolloutRecord, body: CheckpointBody) -> CheckpointRecord {
    let checkpoint_id = body
        .checkpoint_id
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("ckpt_{}", &Uuid::new_v4().to_string()[..12]));
    let mut metadata = body.metadata;
    let state = record.session.get_state();
    metadata.insert("step".to_string(), json!(state.step));
    metadata.insert("total_reward".to_string(), json!(record.total_reward));
    metadata.insert("env_total_reward".to_string(), json!(record.env_total_reward));
    metadata.insert("player_pos".to_string(), json!(state.player_pos));
    metadata.insert("achievements".to_string(), json!(unlocked_achievements(record)));
    metadata.insert("inventory".to_string(), inventory_payload(&state.inventory));

    let checkpoint = CheckpointRecord {
        checkpoint_id: checkpoint_id.clone(),
        rollout_id: record.rollout_id.clone(),
        label: body.label.filter(|value| !value.trim().is_empty()),
        source: body.source.filter(|value| !value.trim().is_empty()),
        actor_ids: body.actor_ids,
        created_at: now_iso(),
        metadata,
        checkpoint_bytes: serde_json::to_vec(&SaveData::from_session(&record.session, None))
            .expect("serialize checkpoint"),
    };
    record
        .checkpoints
        .insert(checkpoint_id.clone(), checkpoint.clone());
    checkpoint
}

fn checkpoint_descriptor(checkpoint: &CheckpointRecord) -> Value {
    json!({
        "checkpoint_id": checkpoint.checkpoint_id,
        "rollout_id": checkpoint.rollout_id,
        "label": checkpoint.label,
        "source": checkpoint.source,
        "actor_ids": checkpoint.actor_ids,
        "created_at": checkpoint.created_at,
        "metadata": checkpoint.metadata,
    })
}

fn rollout_resource(record: &RolloutRecord) -> Value {
    let state = record.session.get_state();
    json!({
        "rollout_id": record.rollout_id,
        "trial_id": record.trial_id,
        "status": record.status,
        "parent_rollout_id": record.parent_rollout_id,
        "trace_correlation_id": record.trace_correlation_id,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "metadata": {
            "seed": record.seed,
            "step_count": state.step,
            "total_reward": record.total_reward,
            "env_total_reward": record.env_total_reward,
            "planner_mode": record.planner_mode,
            "completed_waypoints": record.completed_waypoints,
            "current_waypoint_index": record.current_waypoint_index,
            "planner_failure_code": record.planner_failure_code,
            "player_pos": state.player_pos,
            "achievements": unlocked_achievements(record),
            "inventory": inventory_payload(&state.inventory),
            "checkpoints_available": record.checkpoints.keys().cloned().collect::<Vec<_>>(),
        }
    })
}

fn rollout_trace(record: &RolloutRecord) -> Value {
    json!({
        "schema_version": "4.0",
        "event_history": record.trajectory.iter().map(|item| json!({
            "step_idx": item.step_idx,
            "event_type": "env_step",
            "actions": item.actions,
            "reward": item.reward,
            "total_reward": item.total_reward,
            "env_reward": item.env_reward,
            "env_total_reward": item.env_total_reward,
            "done": item.done,
            "player_pos": item.player_pos,
            "achievement_count": item.achievements.len(),
            "achievements": item.achievements,
            "newly_unlocked": item.newly_unlocked,
            "inventory": item.inventory,
            "planner_mode": record.planner_mode,
            "current_waypoint_index": item.current_waypoint_index,
        })).collect::<Vec<_>>(),
        "markov_blanket_message_history": record.trajectory.iter().map(|item| json!({
            "role": "assistant",
            "step_idx": item.step_idx,
            "content": json!({"actions": item.actions}).to_string(),
        })).collect::<Vec<_>>(),
        "inference": {
            "turns": decision_turns_json(record),
        },
        "metadata": {
            "trace_correlation_id": record.trace_correlation_id,
            "trial_id": record.trial_id,
            "seed": record.seed,
            "step_count": record.session.get_state().step,
            "total_reward": record.total_reward,
            "env_total_reward": record.env_total_reward,
            "planner_mode": record.planner_mode,
            "completed_waypoints": record.completed_waypoints,
            "current_waypoint_index": record.current_waypoint_index,
            "checkpoints_available": record.checkpoints.keys().cloned().collect::<Vec<_>>(),
            "inference_error_count": record.inference_error_count,
            "last_inference_error": record.last_inference_error,
            "llm_call_count": record.llm_call_count,
            "llm_action_batches": record.llm_action_batches,
        }
    })
}

fn reward_details(record: &RolloutRecord) -> Value {
    let state = record.session.get_state();
    json!({
        "trial_id": record.trial_id,
        "seed": record.seed,
        "step_count": state.step,
        "total_reward": record.total_reward,
        "env_total_reward": record.env_total_reward,
        "policy_version": policy_behavior_version(record),
        "checkpoint_count": record.checkpoints.len(),
        "completed_waypoints": record.completed_waypoints,
        "current_waypoint_index": record.current_waypoint_index,
        "supports_restart_from_seed_checkpoint": true,
        "supports_run_from_seed_checkpoint_until": true,
        "supports_terminate": true,
        "planner_mode": record.planner_mode,
        "planner_waypoint_count": record.waypoints.len(),
        "inference_error_count": record.inference_error_count,
        "last_inference_error": record.last_inference_error,
        "llm_call_count": record.llm_call_count,
        "llm_action_batches": record.llm_action_batches,
        "reward_type": "unique_achievements",
        "player_pos": state.player_pos,
        "achievements": unlocked_achievements(record),
        "inventory": inventory_payload(&state.inventory),
    })
}

fn rollout_response(record: &RolloutRecord) -> Value {
    let resource = rollout_resource(record);
    let metadata = resource
        .get("metadata")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let success_status = if record.status == "failed" { "failed" } else { "success" };
    json!({
        "rollout_id": record.rollout_id,
        "trial_id": record.trial_id,
        "status": record.status,
        "metadata": metadata,
        "trace_correlation_id": record.trace_correlation_id,
        "rollout": resource,
        "reward_info": {
            "outcome_reward": record.total_reward,
            "event_rewards": record.decision_turns.iter().map(|item| item.decision_reward).collect::<Vec<_>>(),
            "outcome_objectives": {
                "reward": record.total_reward,
                "unique_achievements": record.total_reward,
            },
            "details": reward_details(record),
            "planner_used": record.planner_mode.as_deref() == Some("waypoint_planned") && !record.waypoints.is_empty(),
            "planner_failure_code": record.planner_failure_code,
        },
        "artifact": [episode_artifact(record)],
        "trace": rollout_trace(record),
        "inference_url": record.inference_url,
        "success_status": success_status,
        "status_detail": record.last_status_detail,
    })
}

fn policy_behavior_version(record: &RolloutRecord) -> String {
    match record.policy_config.get("policy_version") {
        Some(Value::String(text)) if !text.trim().is_empty() => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        _ => "bootstrap".to_string(),
    }
}

fn policy_model(record: &RolloutRecord) -> String {
    record
        .policy_config
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(DEFAULT_MODEL)
        .to_string()
}

fn policy_route(record: &RolloutRecord) -> String {
    record
        .policy_config
        .get("route")
        .or_else(|| record.policy_config.get("policy_role"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("student")
        .to_string()
}

fn discount_rewards(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let mut out = vec![0.0; rewards.len()];
    let mut running = 0.0f64;
    for (idx, reward) in rewards.iter().enumerate().rev() {
        running = *reward + gamma * running;
        out[idx] = running;
    }
    out
}

fn decision_turns_json(record: &RolloutRecord) -> Vec<Value> {
    let gamma = as_f64(record.env_config.get("gamma"), 0.99);
    let rewards = record
        .decision_turns
        .iter()
        .map(|item| item.decision_reward)
        .collect::<Vec<_>>();
    let returns = discount_rewards(&rewards, gamma);
    record
        .decision_turns
        .iter()
        .zip(returns.iter())
        .map(|(item, return_to_go)| {
            json!({
                "turn_index": item.turn_index,
                "prompt_messages": item.prompt_messages,
                "assistant_text": item.assistant_text,
                "reasoning_text": item.reasoning_text,
                "actions": item.actions,
                "decision_reward": item.decision_reward,
                "return_to_go": return_to_go,
                "episode_return": record.total_reward,
                "reward_before": item.reward_before,
                "reward_after": item.reward_after,
                "env_reward_before": item.env_reward_before,
                "env_reward_after": item.env_reward_after,
                "step_start": item.step_start,
                "step_end": item.step_end,
                "trainable": item.trainable,
                "invalid_parse": item.invalid_parse,
                "behavior_version": item.behavior_version,
                "behavior_model": item.behavior_model,
                "route": item.route,
                "request_id": item.request_id,
                "assistant_sequence_logprob": item.behavior_sequence_logprob,
                "behavior_sequence_logprob": item.behavior_sequence_logprob,
                "usage": item.usage,
                "trace_correlation_id": record.trace_correlation_id,
                "trial_id": record.trial_id,
                "outcome_reward": record.total_reward,
            })
        })
        .collect()
}

fn episode_artifact(record: &RolloutRecord) -> Value {
    let state = record.session.get_state();
    json!({
        "trace_correlation_id": record.trace_correlation_id,
        "trial_id": record.trial_id,
        "rollout_id": record.rollout_id,
        "seed": record.seed,
        "behavior_version": policy_behavior_version(record),
        "behavior_model": policy_model(record),
        "outcome_reward": record.total_reward,
        "env_reward": record.env_total_reward,
        "step_count": state.step,
        "invalid_action_count": record.inference_error_count,
        "turns": decision_turns_json(record),
        "trace": rollout_trace(record),
    })
}

fn as_u32(value: Option<&Value>, default: u32) -> u32 {
    value
        .and_then(|value| match value {
            Value::Number(number) => number.as_u64().map(|item| item as u32),
            Value::String(text) => text.parse::<u32>().ok(),
            _ => None,
        })
        .unwrap_or(default)
}

fn as_u64(value: Option<&Value>, default: u64) -> u64 {
    value
        .and_then(|value| match value {
            Value::Number(number) => number.as_u64(),
            Value::String(text) => text.parse::<u64>().ok(),
            _ => None,
        })
        .unwrap_or(default)
}

fn as_i64(value: Option<&Value>, default: i64) -> i64 {
    value
        .and_then(|value| match value {
            Value::Number(number) => number.as_i64(),
            Value::String(text) => text.parse::<i64>().ok(),
            _ => None,
        })
        .unwrap_or(default)
}

fn as_i32(value: Option<&Value>, default: i32) -> i32 {
    as_i64(value, default as i64) as i32
}

fn as_bool(value: Option<&Value>, default: bool) -> bool {
    value
        .and_then(|value| match value {
            Value::Bool(flag) => Some(*flag),
            Value::String(text) => match text.as_str() {
                "true" | "1" => Some(true),
                "false" | "0" => Some(false),
                _ => None,
            },
            _ => None,
        })
        .unwrap_or(default)
}

fn as_f64(value: Option<&Value>, default: f64) -> f64 {
    value
        .and_then(|value| match value {
            Value::Number(number) => number.as_f64(),
            Value::String(text) => text.parse::<f64>().ok(),
            _ => None,
        })
        .unwrap_or(default)
}
