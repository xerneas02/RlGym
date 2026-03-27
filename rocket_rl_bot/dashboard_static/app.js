const VIEW_TITLES = {
  launch: "Lancement",
  states: "State",
  rewards: "Reward",
  replays: "Entrainement Replays",
  models: "Runs & Models",
  jobs: "Jobs",
};

const CHART_COLORS = ["#bb5a2a", "#1f7a78", "#d6a43d", "#6b8e23", "#914f95", "#a84b5f"];
const PREVIEW_FIELD_X = 4096;
const PREVIEW_FIELD_Y = 5120;
const PREVIEW_GOAL_DEPTH = 880;
const PREVIEW_GOAL_HALF_WIDTH = 893;
const PREVIEW_CORNER_OFFSET = 1152;
const PREVIEW_SIDE_WALL_Y = PREVIEW_FIELD_Y - PREVIEW_CORNER_OFFSET;
const PREVIEW_GOAL_BACK_Y = PREVIEW_FIELD_Y + PREVIEW_GOAL_DEPTH;
const PREVIEW_TOTAL_HALF_Y = PREVIEW_GOAL_BACK_Y;

const appState = {
  bootstrap: null,
  docIndex: {},
  currentView: "launch",
  launch: null,
  rewardEditor: null,
  selectedStateId: null,
  stateDetail: null,
  stateEditor: null,
  statePreviewExpanded: false,
  selectedRun: null,
  runDetail: null,
  selectedJob: null,
  jobDetail: null,
  jobLog: "",
  refreshing: false,
};

function clone(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function slugify(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-zA-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase() || "modele";
}

function getByPath(source, path) {
  if (!source) return undefined;
  return path.split(".").reduce((current, key) => (current == null ? undefined : current[key]), source);
}

function setByPath(source, path, value) {
  const parts = path.split(".");
  let current = source;
  for (let index = 0; index < parts.length - 1; index += 1) {
    const key = parts[index];
    if (!current[key] || typeof current[key] !== "object") current[key] = {};
    current = current[key];
  }
  current[parts[parts.length - 1]] = value;
}

function toNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function toBool(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return value === "true";
  return Boolean(value);
}

function unique(values) {
  return Array.from(new Set((values || []).filter((value) => value !== undefined && value !== null && value !== "")));
}

function formatNumber(value, digits = 3) {
  if (value == null || value === "") return "-";
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return String(value);
  return parsed.toLocaleString("fr-FR", { maximumFractionDigits: digits });
}

function formatDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("fr-FR");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/\n/g, "&#10;");
}

async function fetchJSON(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${response.status} ${response.statusText}`);
  return data;
}

function buildDocIndex(parameterDocs) {
  const docs = {};
  Object.values(parameterDocs || {}).forEach((items) => {
    items.forEach((item) => {
      docs[item.key] = item;
    });
  });
  return docs;
}

function docByKey(key) {
  return appState.docIndex[key] || null;
}

function helpTitle(doc) {
  if (!doc) return "";
  return `${doc.label}\n\n${doc.description}\n\nExemple: ${doc.example}\nConseil: ${doc.advice}`;
}

function renderHelpButton(doc) {
  if (!doc) return "";
  return `<button type="button" class="help" title="${escapeAttr(helpTitle(doc))}">?</button>`;
}

function emptyCard(message) {
  return `<div class="card empty-card"><h3>Rien a afficher</h3><p>${escapeHtml(message)}</p></div>`;
}

function selectedMode() {
  const modeId = appState.launch?.modeId || "duel";
  return (appState.bootstrap?.mode_presets || []).find((item) => item.id === modeId) || null;
}

function teamSizeFromMode() {
  const mode = selectedMode();
  return toNumber(getByPath(mode, "config_overrides.config.environment.team_size"), 1);
}

function opponentChoices() {
  return appState.bootstrap?.opponent_choices || [];
}

function modelCatalog() {
  return appState.bootstrap?.model_catalog || [];
}

function rewardProfiles() {
  return appState.bootstrap?.reward_profiles?.profiles || [];
}

function profileById(profileId) {
  return rewardProfiles().find((item) => item.id === profileId) || null;
}

function currentStateRewardDraft() {
  return clone(profileById(appState.stateDetail?.reward_profile)?.weights || {});
}

function selectedRewardDraft() {
  return appState.rewardEditor?.draftWeights || {};
}

function selectedRewardProfileName() {
  if (!appState.rewardEditor) return "";
  if (appState.rewardEditor.profileMode === "new") return appState.rewardEditor.newProfileName || "Nouveau profil";
  const profile = rewardProfiles().find((item) => item.id === appState.rewardEditor.selectedProfileId);
  return profile?.name || appState.rewardEditor.selectedProfileId || "Profil";
}

function defaultRewardWeight(rewardId) {
  const activeProfile = rewardProfiles().find((item) => item.id === (appState.bootstrap?.reward_profiles?.active_profile || "default"));
  const fallbackWeights = activeProfile?.weights || appState.bootstrap?.rewards?.weights || {};
  const value = fallbackWeights[rewardId];
  return Number.isFinite(Number(value)) ? Number(value) : 0.05;
}

function seedOpponentWeights(bootstrap) {
  const configured = new Map((getByPath(bootstrap, "config.training.opponent_mix.opponent_pool") || []).map((entry) => [String(entry.name), Number(entry.weight || 0)]));
  return opponentChoices().map((choice) => ({
    name: choice.id,
    label: choice.label,
    weight: Number.isFinite(configured.get(choice.id)) ? configured.get(choice.id) : 0,
  }));
}

function ensureOpponentWeights() {
  if (!appState.launch) return [];
  const existing = new Map((appState.launch.opponents || []).map((entry) => [String(entry.name), entry]));
  appState.launch.opponents = opponentChoices().map((choice) => {
    const current = existing.get(choice.id);
    return {
      name: choice.id,
      label: choice.label,
      weight: Number.isFinite(Number(current?.weight)) ? Number(current.weight) : 0,
    };
  });
  return appState.launch.opponents;
}

function selectedModelRecord() {
  const selectedModelId = appState.launch?.selectedModelId || "new";
  return modelCatalog().find((item) => item.id === selectedModelId) || modelCatalog()[0] || { id: "new", name: "Nouveau modele", run_name_prefix: "ppo_dashboard" };
}

function computedModelName() {
  if (!appState.launch) return "modele";
  if (appState.launch.selectedModelId === "new") return appState.launch.newModelName || "Nouveau modele";
  return selectedModelRecord().name || "Modele";
}

function computedRunPrefix() {
  if (!appState.launch) return "ppo_dashboard";
  if (appState.launch.selectedModelId === "new") return slugify(appState.launch.newModelName || "ppo_dashboard");
  return slugify(selectedModelRecord().run_name_prefix || selectedModelRecord().name || "ppo_dashboard");
}

function seedClientState(bootstrap) {
  const activeProfileId = bootstrap.reward_profiles?.active_profile || bootstrap.reward_profiles?.profiles?.[0]?.id || "default";
  const activeProfile = (bootstrap.reward_profiles?.profiles || []).find((item) => item.id === activeProfileId) || bootstrap.reward_profiles?.profiles?.[0] || { id: "default", name: "Default", weights: clone(bootstrap.rewards?.weights || {}) };
  appState.launch = {
    modeId: bootstrap.mode_presets?.[0]?.id || "duel",
    selectedModelId: "new",
    newModelName: "",
    notes: "",
    config: clone(bootstrap.config),
    opponents: seedOpponentWeights(bootstrap),
  };
  appState.rewardEditor = {
    profileMode: "existing",
    selectedProfileId: activeProfile.id,
    newProfileName: "",
    draftWeights: clone(activeProfile.weights || bootstrap.rewards?.weights || {}),
  };
  ensureOpponentWeights();
}
function renderConfigField(path) {
  const doc = docByKey(path);
  if (!doc) return "";
  const value = getByPath(appState.launch.config, path);
  const label = `<label>${escapeHtml(doc.label)} ${renderHelpButton(doc)}</label>`;
  if (doc.type === "bool") {
    return `
      <div class="field">
        ${label}
        <select data-launch-config="${escapeAttr(path)}">
          <option value="true" ${toBool(value) ? "selected" : ""}>true</option>
          <option value="false" ${!toBool(value) ? "selected" : ""}>false</option>
        </select>
      </div>
    `;
  }
  const inputType = doc.type === "int" || doc.type === "float" ? "number" : "text";
  const step = doc.type === "int" ? "1" : doc.type === "float" ? "any" : "";
  return `
    <div class="field">
      ${label}
      <input type="${inputType}" step="${step}" value="${escapeAttr(value ?? "")}" data-launch-config="${escapeAttr(path)}">
    </div>
  `;
}

function renderModelSelect(selectId, newNameId, selectedId, newName) {
  const options = modelCatalog().map((item) => `<option value="${escapeAttr(item.id)}" ${item.id === selectedId ? "selected" : ""}>${escapeHtml(item.name)}</option>`).join("");
  return `
    <div class="field">
      <label>Modele</label>
      <select id="${selectId}">${options}</select>
    </div>
    ${selectedId === "new" ? `<div class="field"><label>Nouveau nom de modele</label><input id="${newNameId}" type="text" value="${escapeAttr(newName || "")}" placeholder="Mon modele duel v4"></div>` : ""}
  `;
}

function renderOpponentRows() {
  return ensureOpponentWeights().map((row) => `
    <tr>
      <td><strong>${escapeHtml(row.label || row.name)}</strong><br><span class="subtle">${escapeHtml(row.name)}</span></td>
      <td><input class="weight-input" type="number" step="any" min="0" value="${escapeAttr(row.weight)}" data-opponent-name="${escapeAttr(row.name)}"></td>
      <td class="subtle">0 = desactive, poids relatif sinon.</td>
    </tr>
  `).join("");
}

function rewardRowHtml(weights, prefix) {
  return (appState.bootstrap?.reward_catalog || []).map((reward) => {
    const currentWeight = Number(weights[reward.id] ?? 0);
    const enabled = Math.abs(currentWeight) > 0;
    return `
      <tr class="${enabled ? "" : "row-disabled"}">
        <td><input type="checkbox" data-${prefix}-reward-enabled="${escapeAttr(reward.id)}" ${enabled ? "checked" : ""}></td>
        <td><strong>${escapeHtml(reward.name)}</strong><br><span class="subtle">${escapeHtml(reward.id)}</span></td>
        <td><input type="number" step="any" value="${escapeAttr(currentWeight)}" data-${prefix}-reward-weight="${escapeAttr(reward.id)}" ${enabled ? "" : "disabled"}></td>
        <td>${escapeHtml(reward.description || "")}</td>
        <td>${escapeHtml(reward.advice || "")}</td>
      </tr>
    `;
  }).join("");
}

function renderLaunch() {
  if (!appState.bootstrap || !appState.launch) return emptyCard("Bootstrap indisponible.");
  const modeCards = (appState.bootstrap.mode_presets || []).map((mode) => `
    <button type="button" class="mode-card mode-select ${mode.id === appState.launch.modeId ? "active" : ""}" data-mode-id="${escapeAttr(mode.id)}">
      <div class="inline"><h4>${escapeHtml(mode.name)}</h4><span class="badge ${escapeAttr(mode.status || "ready")}">${escapeHtml(mode.status || "ready")}</span></div>
      <p>${escapeHtml(mode.description || "")}</p>
    </button>
  `).join("");
  return `
    <div class="card">
      <div class="section-header"><div><h3>Mode d'entrainement</h3><p>La taille des equipes est deduite uniquement du mode selectionne.</p></div></div>
      <div class="mode-grid">${modeCards}</div>
    </div>

    <div class="card">
      <div class="section-header"><div><h3>Lancer un entrainement PPO</h3><p>Menu simplifie, coherent et centre sur le mode, le modele et les adversaires.</p></div><div class="inline"><button id="launch-train-btn" class="primary-btn">Lancer l'entrainement</button></div></div>
      <div class="form-grid launch-grid">
        ${renderModelSelect("launch-model-select", "launch-new-model-name", appState.launch.selectedModelId, appState.launch.newModelName)}
        <div class="field"><label>Device ${renderHelpButton(docByKey("project.device"))}</label><select id="launch-device-select">${(appState.bootstrap.device_choices || []).map((value) => `<option value="${escapeAttr(value)}" ${appState.launch.config.project.device === value ? "selected" : ""}>${escapeHtml(value)}</option>`).join("")}</select></div>
        <div class="field seed-field"><label>Seed ${renderHelpButton(docByKey("project.seed"))}</label><div class="seed-input-group"><input id="launch-seed-input" type="number" step="1" value="${escapeAttr(appState.launch.config.project.seed)}"><button type="button" id="launch-random-seed" class="ghost-btn">Generer</button></div></div>
        <div class="field"><label>Run prefix calcule</label><input type="text" value="${escapeAttr(computedRunPrefix())}" disabled></div>
        <div class="field" style="grid-column:1 / -1;"><label>Notes</label><input id="launch-notes" type="text" value="${escapeAttr(appState.launch.notes || "")}" placeholder="Objectif, hypothese, observations..."></div>
        ${["environment.num_envs", "environment.timeout_steps", "environment.no_touch_timeout_steps", "environment.end_on_goal", "environment.goal_reset_to_kickoff", "training.total_steps", "training.rollout_steps", "training.learning_rate", "training.entropy_coef", "training.checkpoint_interval_steps", "evaluation.num_matches", "evaluation.protocol"].map(renderConfigField).join("")}
      </div>
    </div>

    <div class="card">
      <div class="section-header"><div><h3>Gestion des adversaires</h3><p>Definis un poids pour chaque adversaire possible. Le trainer normalise ensuite automatiquement le mix.</p></div></div>
      <div class="table-wrap opponent-matrix-wrap">
        <table class="table opponent-table">
          <thead><tr><th>Adversaire</th><th>Poids</th><th>Usage</th></tr></thead>
          <tbody>${renderOpponentRows()}</tbody>
        </table>
      </div>
    </div>
  `;
}

function renderRewards() {
  if (!appState.bootstrap || !appState.rewardEditor) return emptyCard("Profils de rewards indisponibles.");
  const profileOptions = (rewardProfiles() || []).map((profile) => `<option value="${escapeAttr(profile.id)}" ${profile.id === appState.rewardEditor.selectedProfileId && appState.rewardEditor.profileMode === "existing" ? "selected" : ""}>${escapeHtml(profile.name)}</option>`).join("");
  return `
    <div class="card">
      <div class="section-header"><div><h3>Profils de rewards</h3><p>Selectionne un profil existant ou cree un nouveau profil de rewards.</p></div><div class="inline"><button id="save-reward-profile-btn" class="primary-btn">Sauvegarder le profil</button></div></div>
      <div class="form-grid">
        <div class="field"><label>Profil</label><select id="reward-profile-select"><option value="__new__" ${appState.rewardEditor.profileMode === "new" ? "selected" : ""}>Nouveau profil</option>${profileOptions}</select></div>
        ${appState.rewardEditor.profileMode === "new" ? `<div class="field"><label>Nouveau profil</label><input id="reward-new-profile-name" type="text" value="${escapeAttr(appState.rewardEditor.newProfileName || "")}" placeholder="Mon profil defense"></div>` : `<div class="field"><label>Profil actif</label><input type="text" value="${escapeAttr(selectedRewardProfileName())}" disabled></div>`}
      </div>
    </div>

    <div class="card">
      <div class="section-header"><div><h3>Reward actifs</h3><p>Le tableau a ete deplace ici et chaque ligne peut etre activee ou desactivee visuellement.</p></div></div>
      <div class="table-wrap">
        <table class="table">
          <thead><tr><th>Actif</th><th>Reward</th><th>Poids</th><th>Description</th><th>Conseil</th></tr></thead>
          <tbody>${rewardRowHtml(selectedRewardDraft(), "reward")}</tbody>
        </table>
      </div>
    </div>
  `;
}

function renderReplays() {
  if (!appState.bootstrap || !appState.launch) return emptyCard("Parametres replay indisponibles.");
  const replaySources = appState.bootstrap.replay_sources || [];
  const current = appState.launch.replays || {
    use_all_replays: true,
    max_replays: 0,
    validation_replays: 50,
    sample_fps: 2,
    state_timeout_override: "",
    replay_dirs: replaySources.slice(0, 2).map((item) => item.path),
    extra_replay_dirs: "",
  };
  appState.launch.replays = current;
  return `
    <div class="card">
      <div class="section-header"><div><h3>Entrainement Replays</h3><p>Tous les elements replay ont ete deplaces ici. La taille des equipes suit le mode choisi dans Lancement.</p></div><div class="inline"><button id="launch-pretrain-btn" class="primary-btn">Lancer le pretraining</button></div></div>
      <div class="notice">Mode courant: <strong>${escapeHtml(selectedMode()?.name || "Duel")}</strong> | Team size deduite: <strong>${teamSizeFromMode()}v${teamSizeFromMode()}</strong></div>
      <div class="form-grid" style="margin-top:14px;">
        <div class="field"><label>Utiliser tous les replays</label><select id="replay-use-all"><option value="true" ${current.use_all_replays ? "selected" : ""}>Oui</option><option value="false" ${!current.use_all_replays ? "selected" : ""}>Non</option></select></div>
        <div class="field"><label>Nombre max de replays</label><input id="replay-max-replays" type="number" step="1" value="${escapeAttr(current.max_replays)}" ${current.use_all_replays ? "disabled" : ""}></div>
        <div class="field"><label>Replays de validation</label><input id="replay-validation" type="number" step="1" value="${escapeAttr(current.validation_replays)}"></div>
        <div class="field"><label>Sample FPS</label><input id="replay-sample-fps" type="number" step="any" value="${escapeAttr(current.sample_fps)}"></div>
        <div class="field"><label>Override timeout state</label><input id="replay-timeout" type="number" step="1" value="${escapeAttr(current.state_timeout_override)}" placeholder="vide = garde la config"></div>
      </div>
      <div class="field" style="margin-top:14px;"><label>Dossiers replay detectes</label><div class="mode-grid">${replaySources.length ? replaySources.map((item) => `<label class="mode-card" style="display:block;text-align:left;"><input type="checkbox" data-replay-path="${escapeAttr(item.path)}" ${current.replay_dirs.includes(item.path) ? "checked" : ""}><strong>${escapeHtml(item.name)}</strong><br><span class="subtle">${escapeHtml(item.path)}</span><br><span class="subtle">${formatNumber(item.replay_count, 0)} replay(s), ${formatNumber(item.csv_count, 0)} CSV</span></label>`).join("") : `<div class="notice warning">Aucun dossier replay auto-detecte.</div>`}</div></div>
      <div class="field" style="margin-top:14px;"><label>Chemins replay supplementaires</label><textarea id="replay-extra-dirs" placeholder="Un chemin par ligne ou separes par des virgules">${escapeHtml(current.extra_replay_dirs || "")}</textarea></div>
    </div>
  `;
}
function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(220, Math.floor(canvas.clientWidth || canvas.width || 640));
  const height = Math.max(180, Math.floor(canvas.clientHeight || canvas.height || 260));
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function scenarioToPreview(payload) {
  if (!payload) return null;
  if (payload.preview) return payload.preview;
  const definition = payload.definition || payload;
  const ballPosition = definition.ball?.position || {};
  return {
    ball: [toNumber(ballPosition.x, 0), toNumber(ballPosition.y, 0), toNumber(ballPosition.z, 93)],
    cars: (definition.cars || []).map((car) => ({
      team: toNumber(car.team, 0),
      x: toNumber(car.position?.x, 0),
      y: toNumber(car.position?.y, 0),
      yaw: toNumber(car.yaw?.value, car.team === 0 ? 1.57 : -1.57),
    })),
  };
}

function drawTopDownPreview(canvas, payload) {
  if (!canvas) return;
  const preview = scenarioToPreview(payload);
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const margin = 18;
  const arena = previewArenaRect(width, height, margin);
  const scaleX = (x) => arena.left + ((x + PREVIEW_FIELD_X) / (PREVIEW_FIELD_X * 2)) * arena.width;
  const scaleY = (y) => arena.top + (1 - ((y + PREVIEW_TOTAL_HALF_Y) / (PREVIEW_TOTAL_HALF_Y * 2))) * arena.height;
  const worldPolyline = (points) => points.map(([x, y]) => [scaleX(x), scaleY(y)]);
  const drawPolyline = (points, strokeStyle, lineWidth = 2, closed = false) => {
    if (!points.length) return;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    if (closed) ctx.closePath();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  };
  const fillPolygon = (points, fillStyle) => {
    if (!points.length) return;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
  };
  const soccarOutline = worldPolyline([
    [-PREVIEW_GOAL_HALF_WIDTH, PREVIEW_GOAL_BACK_Y],
    [PREVIEW_GOAL_HALF_WIDTH, PREVIEW_GOAL_BACK_Y],
    [PREVIEW_GOAL_HALF_WIDTH, PREVIEW_FIELD_Y],
    [PREVIEW_FIELD_X - PREVIEW_CORNER_OFFSET, PREVIEW_FIELD_Y],
    [PREVIEW_FIELD_X, PREVIEW_SIDE_WALL_Y],
    [PREVIEW_FIELD_X, -PREVIEW_SIDE_WALL_Y],
    [PREVIEW_FIELD_X - PREVIEW_CORNER_OFFSET, -PREVIEW_FIELD_Y],
    [PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_FIELD_Y],
    [PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_GOAL_BACK_Y],
    [-PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_GOAL_BACK_Y],
    [-PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_FIELD_Y],
    [-(PREVIEW_FIELD_X - PREVIEW_CORNER_OFFSET), -PREVIEW_FIELD_Y],
    [-PREVIEW_FIELD_X, -PREVIEW_SIDE_WALL_Y],
    [-PREVIEW_FIELD_X, PREVIEW_SIDE_WALL_Y],
    [-(PREVIEW_FIELD_X - PREVIEW_CORNER_OFFSET), PREVIEW_FIELD_Y],
    [-PREVIEW_GOAL_HALF_WIDTH, PREVIEW_FIELD_Y],
  ]);
  fillPolygon([[0, 0], [width, 0], [width, height], [0, height]], "#0d2d2b");
  fillPolygon(soccarOutline, "#1d5c3d");
  drawPolyline(soccarOutline, "rgba(235, 243, 233, 0.85)", 2, true);
  drawPolyline(worldPolyline([[0, PREVIEW_FIELD_Y], [0, -PREVIEW_FIELD_Y]]), "rgba(235, 243, 233, 0.6)", 2);
  ctx.beginPath();
  ctx.strokeStyle = "rgba(235, 243, 233, 0.55)";
  ctx.lineWidth = 2;
  ctx.arc(scaleX(0), scaleY(0), Math.abs(scaleY(915) - scaleY(0)), 0, Math.PI * 2);
  ctx.stroke();
  const goalTop = worldPolyline([[-PREVIEW_GOAL_HALF_WIDTH, PREVIEW_GOAL_BACK_Y], [PREVIEW_GOAL_HALF_WIDTH, PREVIEW_GOAL_BACK_Y], [PREVIEW_GOAL_HALF_WIDTH, PREVIEW_FIELD_Y], [-PREVIEW_GOAL_HALF_WIDTH, PREVIEW_FIELD_Y]]);
  const goalBottom = worldPolyline([[-PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_GOAL_BACK_Y], [PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_GOAL_BACK_Y], [PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_FIELD_Y], [-PREVIEW_GOAL_HALF_WIDTH, -PREVIEW_FIELD_Y]]);
  fillPolygon(goalTop, "rgba(255, 151, 80, 0.12)");
  fillPolygon(goalBottom, "rgba(101, 167, 255, 0.12)");
  drawPolyline(goalTop, "rgba(255, 151, 80, 0.75)", 2, true);
  drawPolyline(goalBottom, "rgba(101, 167, 255, 0.75)", 2, true);
  if (!preview) {
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.font = "15px Aptos";
    ctx.fillText("Pas de preview graphique pour ce state.", 22, 34);
    return;
  }
  const definition = payload?.definition || payload || {};
  const drawRangeBox = (range, color) => {
    if (!range?.enabled) return;
    const left = scaleX(range.min_x);
    const right = scaleX(range.max_x);
    const top = scaleY(range.max_y);
    const bottom = scaleY(range.min_y);
    ctx.fillStyle = `${color}22`;
    ctx.strokeStyle = `${color}cc`;
    ctx.lineWidth = 1.5;
    ctx.fillRect(Math.min(left, right), Math.min(top, bottom), Math.abs(right - left), Math.abs(bottom - top));
    ctx.strokeRect(Math.min(left, right), Math.min(top, bottom), Math.abs(right - left), Math.abs(bottom - top));
  };
  drawRangeBox(definition.ball?.position_random, "#ffbf5f");
  (definition.cars || []).forEach((car) => drawRangeBox(car.position_random, Number(car.team) === 0 ? "#6ec9ff" : "#ff985e"));
  const drawArrow = (originX, originY, vecX, vecY, color) => {
    if (Math.hypot(vecX, vecY) < 30) return;
    const fromX = scaleX(originX);
    const fromY = scaleY(originY);
    const toX = scaleX(originX + vecX * 0.08);
    const toY = scaleY(originY + vecY * 0.08);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
    const angle = Math.atan2(toY - fromY, toX - fromX);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - 10 * Math.cos(angle - Math.PI / 6), toY - 10 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(toX - 10 * Math.cos(angle + Math.PI / 6), toY - 10 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  };
  drawArrow(preview.ball?.[0] || 0, preview.ball?.[1] || 0, definition.ball?.velocity?.x || 0, definition.ball?.velocity?.y || 0, "rgba(255, 218, 118, 0.95)");
  ctx.fillStyle = "#f9c45d";
  ctx.beginPath();
  ctx.arc(scaleX(preview.ball?.[0] || 0), scaleY(preview.ball?.[1] || 0), 8, 0, Math.PI * 2);
  ctx.fill();
  (preview.cars || []).forEach((car, index) => {
    const source = definition.cars?.[index];
    drawArrow(car.x || 0, car.y || 0, source?.velocity?.x || 0, source?.velocity?.y || 0, "rgba(165, 255, 214, 0.9)");
    ctx.save();
    ctx.translate(scaleX(car.x || 0), scaleY(car.y || 0));
    ctx.rotate(-(car.yaw || 0) + Math.PI / 2);
    ctx.fillStyle = Number(car.team) === 0 ? "#6ec9ff" : "#ff985e";
    ctx.beginPath();
    ctx.moveTo(0, -11);
    ctx.lineTo(8, 10);
    ctx.lineTo(-8, 10);
    ctx.closePath();
    ctx.fill();
    if (appState.stateEditor?.selectedKind === "car" && appState.stateEditor?.selectedIndex === index) {
      ctx.strokeStyle = "rgba(255, 244, 208, 0.95)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    ctx.restore();
    if (source?.yaw?.enabled) {
      const yawMin = Number(source.yaw.min ?? car.yaw ?? 0);
      const yawMax = Number(source.yaw.max ?? car.yaw ?? 0);
      const centerX = scaleX(car.x || 0);
      const centerY = scaleY(car.y || 0);
      const radius = 32;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.arc(centerX, centerY, radius, -yawMax + Math.PI / 2, -yawMin + Math.PI / 2, false);
      ctx.closePath();
      ctx.fillStyle = Number(car.team) === 0 ? "rgba(110, 201, 255, 0.14)" : "rgba(255, 152, 94, 0.14)";
      ctx.fill();
    }
  });
  if (appState.stateEditor?.selectedKind === "ball") {
    ctx.strokeStyle = "rgba(255, 244, 208, 0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(scaleX(preview.ball?.[0] || 0), scaleY(preview.ball?.[1] || 0), 14, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function defaultVector3(x = 0, y = 0, z = 0) {
  return { x, y, z };
}

function defaultRange3(x = 0, y = 0, z = 0) {
  return { enabled: false, min_x: x, max_x: x, min_y: y, max_y: y, min_z: z, max_z: z };
}

function defaultRange1(value = 0) {
  return { enabled: false, value, min: value, max: value };
}

function defaultCarSpec(team = 0, index = 0) {
  const y = team === 0 ? -2048 : 2048;
  const yaw = team === 0 ? 1.57 : -1.57;
  const prefix = team === 0 ? "blue" : "orange";
  return {
    name: `${prefix}_${index + 1}`,
    team,
    position: defaultVector3(0, y, 17),
    position_random: defaultRange3(0, y, 17),
    velocity: defaultVector3(0, 0, 0),
    yaw: defaultRange1(yaw),
    boost: defaultRange1(0.33),
  };
}

function ensureVector3(vector, defaults) {
  const base = vector && typeof vector === "object" ? vector : {};
  return {
    x: toNumber(base.x, defaults.x),
    y: toNumber(base.y, defaults.y),
    z: toNumber(base.z, defaults.z),
  };
}

function ensureRange3(range, defaults) {
  const base = range && typeof range === "object" ? range : {};
  const minX = toNumber(base.min_x, defaults.x);
  const maxX = toNumber(base.max_x, defaults.x);
  const minY = toNumber(base.min_y, defaults.y);
  const maxY = toNumber(base.max_y, defaults.y);
  const minZ = toNumber(base.min_z, defaults.z);
  const maxZ = toNumber(base.max_z, defaults.z);
  return {
    enabled: toBool(base.enabled),
    min_x: Math.min(minX, maxX),
    max_x: Math.max(minX, maxX),
    min_y: Math.min(minY, maxY),
    max_y: Math.max(minY, maxY),
    min_z: Math.min(minZ, maxZ),
    max_z: Math.max(minZ, maxZ),
  };
}

function ensureRange1(range, defaultValue) {
  const base = range && typeof range === "object" ? range : {};
  const value = toNumber(base.value, defaultValue);
  const min = toNumber(base.min, value);
  const max = toNumber(base.max, value);
  return {
    enabled: toBool(base.enabled),
    value,
    min: Math.min(min, max),
    max: Math.max(min, max),
  };
}

function ensureScenarioDefinition(detail) {
  if (!detail) return null;
  detail.definition = detail.definition && typeof detail.definition === "object" ? detail.definition : {};
  if (detail.kind === "replay_csv") {
    detail.definition.replay_folder = String(detail.definition.replay_folder || "DataState");
    detail.definition.mirrored = toBool(detail.definition.mirrored);
    return detail.definition;
  }
  const definition = detail.definition;
  definition.ball = definition.ball && typeof definition.ball === "object" ? definition.ball : {};
  definition.ball.position = ensureVector3(definition.ball.position, defaultVector3(0, 0, 93));
  definition.ball.position_random = ensureRange3(definition.ball.position_random, definition.ball.position);
  definition.ball.velocity = ensureVector3(definition.ball.velocity, defaultVector3(0, 0, 0));
  definition.ball.angular_velocity = ensureVector3(definition.ball.angular_velocity, defaultVector3(0, 0, 0));
  const cars = Array.isArray(definition.cars) && definition.cars.length ? definition.cars : [defaultCarSpec(0, 0), defaultCarSpec(1, 0)];
  definition.cars = cars.map((car, index) => {
    const base = car && typeof car === "object" ? car : defaultCarSpec(index === 0 ? 0 : 1, index);
    const team = toNumber(base.team, index === 0 ? 0 : 1);
    const positionDefaults = defaultCarSpec(team, index).position;
    return {
      name: String(base.name || `${team === 0 ? "blue" : "orange"}_${index + 1}`),
      team,
      position: ensureVector3(base.position, positionDefaults),
      position_random: ensureRange3(base.position_random, ensureVector3(base.position, positionDefaults)),
      velocity: ensureVector3(base.velocity, defaultVector3(0, 0, 0)),
      yaw: ensureRange1(base.yaw, team === 0 ? 1.57 : -1.57),
      boost: ensureRange1(base.boost, 0.33),
    };
  });
  if (!appState.stateEditor) appState.stateEditor = { selectedKind: "ball", selectedIndex: 0, dragMode: null, status: "Pret" };
  if (appState.stateEditor.selectedKind === "car") {
    appState.stateEditor.selectedIndex = Math.max(0, Math.min(appState.stateEditor.selectedIndex, definition.cars.length - 1));
  }
  return definition;
}

function selectedStateEntity() {
  const definition = ensureScenarioDefinition(appState.stateDetail);
  if (!definition) return null;
  if (!appState.stateEditor) appState.stateEditor = { selectedKind: "ball", selectedIndex: 0, dragMode: null, status: "Pret" };
  if (appState.stateEditor.selectedKind === "car") {
    return { kind: "car", index: appState.stateEditor.selectedIndex, target: definition.cars[appState.stateEditor.selectedIndex] };
  }
  return { kind: "ball", index: 0, target: definition.ball };
}

function renderStateInput(label, path, value, type = "number") {
  const inputType = type === "number" ? 'type="number" step="any"' : 'type="text"';
  return `<div class="field"><label>${escapeHtml(label)}</label><input ${inputType} value="${escapeAttr(value)}" data-state-path="${escapeAttr(path)}" data-state-type="${escapeAttr(type)}"></div>`;
}

function renderStateToggle(label, path, checked) {
  return `<label class="toggle-chip"><input type="checkbox" ${checked ? "checked" : ""} data-state-path="${escapeAttr(path)}" data-state-type="bool"><span>${escapeHtml(label)}</span></label>`;
}

function renderStateTeamToggle(path, value) {
  return `
    <div class="field">
      <label>Equipe</label>
      <div class="entity-strip">
        <button type="button" class="chip-btn ${Number(value) === 0 ? "active" : ""}" data-state-team="${escapeAttr(path)}" data-state-team-value="0">Bleu</button>
        <button type="button" class="chip-btn ${Number(value) === 1 ? "active" : ""}" data-state-team="${escapeAttr(path)}" data-state-team-value="1">Orange</button>
      </div>
    </div>
  `;
}

function renderStateEntityTabs(detail) {
  const definition = ensureScenarioDefinition(detail);
  if (!definition || detail.kind === "replay_csv") return "";
  const items = [
    `<button type="button" class="chip-btn state-entity-select ${appState.stateEditor?.selectedKind === "ball" ? "active" : ""}" data-state-select-kind="ball" data-state-select-index="0">Balle</button>`,
    ...definition.cars.map((car, index) => `<button type="button" class="chip-btn state-entity-select ${appState.stateEditor?.selectedKind === "car" && appState.stateEditor?.selectedIndex === index ? "active" : ""}" data-state-select-kind="car" data-state-select-index="${index}">${escapeHtml(car.name)}</button>`),
  ];
  return `<div class="entity-strip">${items.join("")}</div>`;
}

function renderStateScenarioEditor(detail) {
  const entity = selectedStateEntity();
  if (!entity) return "";
  const target = entity.target;
  const basePath = entity.kind === "ball" ? "definition.ball" : `definition.cars.${entity.index}`;
  const commonFields = [
    renderStateInput("pos_x", `${basePath}.position.x`, target.position.x),
    renderStateInput("pos_y", `${basePath}.position.y`, target.position.y),
    renderStateInput("pos_z", `${basePath}.position.z`, target.position.z),
    renderStateInput("vel_x", `${basePath}.velocity.x`, target.velocity.x),
    renderStateInput("vel_y", `${basePath}.velocity.y`, target.velocity.y),
    renderStateInput("vel_z", `${basePath}.velocity.z`, target.velocity.z),
    renderStateInput("rand_x_min", `${basePath}.position_random.min_x`, target.position_random.min_x),
    renderStateInput("rand_x_max", `${basePath}.position_random.max_x`, target.position_random.max_x),
    renderStateInput("rand_y_min", `${basePath}.position_random.min_y`, target.position_random.min_y),
    renderStateInput("rand_y_max", `${basePath}.position_random.max_y`, target.position_random.max_y),
    renderStateInput("rand_z_min", `${basePath}.position_random.min_z`, target.position_random.min_z),
    renderStateInput("rand_z_max", `${basePath}.position_random.max_z`, target.position_random.max_z),
  ];
  const specificFields = entity.kind === "ball"
    ? [
        renderStateInput("ang_vel_x", `${basePath}.angular_velocity.x`, target.angular_velocity.x),
        renderStateInput("ang_vel_y", `${basePath}.angular_velocity.y`, target.angular_velocity.y),
        renderStateInput("ang_vel_z", `${basePath}.angular_velocity.z`, target.angular_velocity.z),
      ]
    : [
        renderStateInput("Nom", `${basePath}.name`, target.name, "text"),
        renderStateTeamToggle(`${basePath}.team`, target.team),
        renderStateInput("yaw", `${basePath}.yaw.value`, target.yaw.value),
        renderStateInput("yaw_min", `${basePath}.yaw.min`, target.yaw.min),
        renderStateInput("yaw_max", `${basePath}.yaw.max`, target.yaw.max),
        renderStateInput("boost", `${basePath}.boost.value`, target.boost.value),
        renderStateInput("boost_min", `${basePath}.boost.min`, target.boost.min),
        renderStateInput("boost_max", `${basePath}.boost.max`, target.boost.max),
      ];
  const toggles = [renderStateToggle("Random position", `${basePath}.position_random.enabled`, target.position_random.enabled)];
  if (entity.kind === "car") toggles.push(renderStateToggle("Random yaw", `${basePath}.yaw.enabled`, target.yaw.enabled));
  return `
    <div class="state-editor-layout">
      <div class="canvas-wrap state-canvas-panel ${appState.statePreviewExpanded ? "expanded" : ""}">
        <div class="section-header" style="margin-bottom:10px;">
          <div><h3>Preview Terrain</h3><p>Edition visuelle proche du scenario editor.</p></div>
          <div class="inline"><button type="button" id="toggle-state-preview-size" class="ghost-btn">${appState.statePreviewExpanded ? "Reduire" : "Agrandir"}</button></div>
        </div>
        <canvas id="state-detail-preview" class="preview-canvas"></canvas>
        <div class="notice" style="margin-top:12px;">Clic puis glisser: position. <code>Ctrl</code> + glisser: vitesse. <code>Alt</code> + glisser sur une voiture: angle. Clic droit + glisser: zone random. <code>Alt</code> + clic droit: plage d'angle.</div>
        <div class="subtle" style="margin-top:10px;">Statut: ${escapeHtml(appState.stateEditor?.status || "Pret")}</div>
        ${renderStateEntityTabs(detail)}
      </div>
      <div class="card state-editor-panel">
        <div class="section-header"><div><h3>${escapeHtml(entity.kind === "ball" ? "Edition de la balle" : `Edition de ${target.name}`)}</h3><p>Editeur structure inspire du scenario editor, sans zone JSON brute.</p></div><div class="inline"><button type="button" id="state-add-blue-car" class="ghost-btn">Ajouter bleu</button><button type="button" id="state-add-orange-car" class="ghost-btn">Ajouter orange</button>${entity.kind === "car" ? `<button type="button" id="state-delete-car" class="danger-btn">Supprimer cette voiture</button>` : ""}</div></div>
        <div class="toggle-row">${toggles.join("")}</div>
        <div class="form-grid compact state-editor-grid">${commonFields.concat(specificFields).join("")}</div>
      </div>
    </div>
  `;
}

function renderReplayStateEditor(detail) {
  const definition = ensureScenarioDefinition(detail);
  return `
    <div class="card">
      <div class="section-header"><div><h3>Configuration replay CSV</h3><p>Ce state pointe vers des snapshots replay convertis en CSV.</p></div></div>
      <div class="form-grid compact">
        ${renderStateInput("Dossier replay", "definition.replay_folder", definition.replay_folder || "DataState", "text")}
        <div class="field"><label>Miroir</label><select data-state-path="definition.mirrored" data-state-type="bool-string"><option value="false" ${!definition.mirrored ? "selected" : ""}>false</option><option value="true" ${definition.mirrored ? "selected" : ""}>true</option></select></div>
      </div>
      <div class="notice" style="margin-top:14px;">La preview 2D n'est pas disponible pour un dossier replay, mais l'option miroir reste bien editable ici.</div>
    </div>
  `;
}

function previewArenaRect(width, height, margin = 18) {
  const availableWidth = Math.max(1, width - margin * 2);
  const availableHeight = Math.max(1, height - margin * 2);
  const arenaAspect = (PREVIEW_FIELD_X * 2) / (PREVIEW_TOTAL_HALF_Y * 2);
  const availableAspect = availableWidth / availableHeight;
  let arenaWidth;
  let arenaHeight;
  if (availableAspect > arenaAspect) {
    arenaHeight = availableHeight;
    arenaWidth = arenaHeight * arenaAspect;
  } else {
    arenaWidth = availableWidth;
    arenaHeight = arenaWidth / arenaAspect;
  }
  return {
    left: (width - arenaWidth) / 2,
    top: (height - arenaHeight) / 2,
    width: arenaWidth,
    height: arenaHeight,
  };
}

function previewPoint(canvas, worldX, worldY) {
  const width = canvas.clientWidth || canvas.width || 640;
  const height = canvas.clientHeight || canvas.height || 320;
  const arena = previewArenaRect(width, height, 18);
  return {
    x: arena.left + ((worldX + PREVIEW_FIELD_X) / (PREVIEW_FIELD_X * 2)) * arena.width,
    y: arena.top + (1 - ((worldY + PREVIEW_TOTAL_HALF_Y) / (PREVIEW_TOTAL_HALF_Y * 2))) * arena.height,
  };
}

function previewWorld(canvas, localX, localY) {
  const width = canvas.clientWidth || canvas.width || 640;
  const height = canvas.clientHeight || canvas.height || 320;
  const arena = previewArenaRect(width, height, 18);
  const usableWidth = Math.max(1, arena.width);
  const usableHeight = Math.max(1, arena.height);
  const x = ((localX - arena.left) / usableWidth) * (PREVIEW_FIELD_X * 2) - PREVIEW_FIELD_X;
  const y = (1 - ((localY - arena.top) / usableHeight)) * (PREVIEW_TOTAL_HALF_Y * 2) - PREVIEW_TOTAL_HALF_Y;
  return {
    x: Math.max(-PREVIEW_FIELD_X, Math.min(PREVIEW_FIELD_X, x)),
    y: Math.max(-PREVIEW_GOAL_BACK_Y, Math.min(PREVIEW_GOAL_BACK_Y, y)),
  };
}

function pickPreviewEntity(canvas, payload, localX, localY) {
  const preview = scenarioToPreview(payload);
  if (!preview) return null;
  const items = [{ kind: "ball", index: 0, x: preview.ball?.[0] || 0, y: preview.ball?.[1] || 0 }]
    .concat((preview.cars || []).map((car, index) => ({ kind: "car", index, x: car.x || 0, y: car.y || 0 })));
  let best = null;
  items.forEach((item) => {
    const point = previewPoint(canvas, item.x, item.y);
    const distance = Math.hypot(point.x - localX, point.y - localY);
    if (!best || distance < best.distance) best = { ...item, distance };
  });
  return best && best.distance <= 30 ? best : null;
}

function captureScrollState() {
  return {
    windowY: window.scrollY,
    panels: Array.from(document.querySelectorAll(".view.active .list-panel, .view.active .detail-panel")).map((node) => node.scrollTop),
  };
}

function restoreScrollState(snapshot) {
  if (!snapshot) return;
  window.scrollTo(0, snapshot.windowY || 0);
  Array.from(document.querySelectorAll(".view.active .list-panel, .view.active .detail-panel")).forEach((node, index) => {
    if (snapshot.panels[index] != null) node.scrollTop = snapshot.panels[index];
  });
}

function isEditingField() {
  const active = document.activeElement;
  return Boolean(active && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName));
}

function numericKeys(rows) {
  if (!rows || !rows.length) return [];
  const counts = {};
  rows.forEach((row) => {
    Object.entries(row).forEach(([key, value]) => {
      if (Number.isFinite(Number(value))) counts[key] = (counts[key] || 0) + 1;
    });
  });
  return Object.keys(counts).filter((key) => key !== "epoch").sort();
}

function pickSeries(rows, preferred) {
  const available = numericKeys(rows);
  const picked = preferred.filter((key) => available.includes(key));
  return (picked.length ? picked : available.filter((key) => key !== "step")).slice(0, 3);
}

function drawLineChart(canvas, rows, seriesKeys, xKey = "step") {
  if (!canvas) return;
  const { ctx, width, height } = resizeCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fillRect(0, 0, width, height);
  const filteredRows = (rows || []).filter((row) => seriesKeys.some((key) => Number.isFinite(Number(row[key]))));
  if (!filteredRows.length || !seriesKeys.length) {
    ctx.fillStyle = "#6f6154";
    ctx.font = "14px Aptos";
    ctx.fillText("Pas assez de donnees pour tracer une courbe.", 18, 28);
    return;
  }
  const padding = { top: 24, right: 24, bottom: 28, left: 42 };
  const usableWidth = width - padding.left - padding.right;
  const usableHeight = height - padding.top - padding.bottom;
  const xValues = filteredRows.map((row, index) => Number.isFinite(Number(row[xKey])) ? Number(row[xKey]) : index);
  const yValues = filteredRows.flatMap((row) => seriesKeys.map((key) => Number(row[key])).filter((value) => Number.isFinite(value)));
  let minY = Math.min(...yValues);
  let maxY = Math.max(...yValues);
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const minX = Math.min(...xValues);
  const maxX = Math.max(...xValues);
  const xSpan = maxX - minX || 1;
  const ySpan = maxY - minY || 1;
  ctx.strokeStyle = "rgba(71,54,36,0.18)";
  for (let index = 0; index <= 4; index += 1) {
    const y = padding.top + (usableHeight * index) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#5c4c3a";
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();
  seriesKeys.forEach((key, index) => {
    ctx.strokeStyle = CHART_COLORS[index % CHART_COLORS.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    filteredRows.forEach((row, rowIndex) => {
      const xValue = Number.isFinite(Number(row[xKey])) ? Number(row[xKey]) : rowIndex;
      const yValue = Number(row[key]);
      const x = padding.left + ((xValue - minX) / xSpan) * usableWidth;
      const y = padding.top + (1 - (yValue - minY) / ySpan) * usableHeight;
      if (rowIndex === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = CHART_COLORS[index % CHART_COLORS.length];
    ctx.fillRect(padding.left + index * 140, 6, 12, 12);
    ctx.fillStyle = "#24180d";
    ctx.font = "12px Aptos";
    ctx.fillText(key, padding.left + 18 + index * 140, 16);
  });
}

function stateCompatibility(state) {
  const supported = state.supported_team_sizes || [1, 2, 3];
  return supported.includes(teamSizeFromMode());
}

function stateRewardProfileOptions(selectedId) {
  return rewardProfiles().map((profile) => `<option value="${escapeAttr(profile.id)}" ${profile.id === selectedId ? "selected" : ""}>${escapeHtml(profile.name)}</option>`).join("");
}

function renderStates() {
  if (!appState.bootstrap) return emptyCard("Catalogue de states indisponible.");
  const states = appState.bootstrap.state_catalog || [];
  const rows = states.map((state) => {
    const compatible = stateCompatibility(state);
    return `
      <tr class="state-row ${compatible ? "" : "row-disabled"} ${appState.selectedStateId === state.id ? "active" : ""}" data-state-id="${escapeAttr(state.id)}">
        <td><input type="checkbox" ${state.enabled ? "checked" : ""} data-state-toggle="${escapeAttr(state.id)}" ${compatible ? "" : "disabled"}></td>
        <td><strong>${escapeHtml(state.name)}</strong><br><span class="subtle">${escapeHtml(state.id)}</span></td>
        <td>${escapeHtml(state.category || "custom")}</td>
        <td>${escapeHtml((state.supported_team_sizes || [1, 2, 3]).join(", "))}</td>
        <td>${escapeHtml(state.reward_profile || "default")}</td>
      </tr>
    `;
  }).join("");

  if (!appState.stateDetail) {
    return `
      <div class="state-workspace">
        <div class="card list-panel">
          <div class="section-header"><div><h3>States JSON</h3><p>Tous les states du dashboard sont maintenant pilotes par JSON.</p></div></div>
          <div class="table-wrap"><table class="table"><thead><tr><th>Actif</th><th>State</th><th>Categorie</th><th>Players</th><th>Reward profile</th></tr></thead><tbody>${rows}</tbody></table></div>
        </div>
        <div class="detail-panel">${emptyCard("Clique sur un state pour ouvrir sa fiche detail et son editeur structure.")}</div>
      </div>
    `;
  }

  const detail = appState.stateDetail;
  ensureScenarioDefinition(detail);
  const editorBlock = detail.kind === "replay_csv" ? renderReplayStateEditor(detail) : renderStateScenarioEditor(detail);
  return `
    <div class="state-workspace">
      <div class="card list-panel">
        <div class="section-header"><div><h3>States JSON</h3><p>Les states incompatibles avec le mode courant sont grises et non reactibles, mais restent editables.</p></div></div>
        <div class="table-wrap"><table class="table"><thead><tr><th>Actif</th><th>State</th><th>Categorie</th><th>Players</th><th>Reward profile</th></tr></thead><tbody>${rows}</tbody></table></div>
      </div>
      <div class="detail-panel">
        <div class="card">
          <div class="section-header"><div><h3>${escapeHtml(detail.name || detail.id)}</h3><p>${escapeHtml(detail.description || "")}</p></div><div class="inline"><button id="state-mirror-btn" class="ghost-btn">Miroir</button><button id="save-state-btn" class="primary-btn">Sauvegarder</button></div></div>
          <div class="form-grid">
            <div class="field"><label>Nom</label><input id="state-name-input" type="text" value="${escapeAttr(detail.name || detail.id)}"></div>
            <div class="field"><label>Categorie</label><input id="state-category-input" type="text" value="${escapeAttr(detail.category || "custom")}"></div>
            <div class="field"><label>Reward profile lie</label><select id="state-reward-profile-select">${stateRewardProfileOptions(detail.reward_profile || rewardProfiles()[0]?.id || "default")}</select></div>
            <div class="field"><label>Teams supportees</label><input id="state-supported-input" type="text" value="${escapeAttr((detail.supported_team_sizes || [1, 2, 3]).join(","))}" placeholder="1,2,3"></div>
            <div class="field" style="grid-column:1 / -1;"><label>Description</label><input id="state-description-input" type="text" value="${escapeAttr(detail.description || "")}"></div>
          </div>
        </div>
        ${editorBlock}
      </div>
    </div>
  `;
}
function buildStateCurriculumOverride() {
  const enabledStates = (appState.bootstrap?.state_catalog || []).filter((state) => state.enabled && stateCompatibility(state));
  const curriculum = clone(appState.bootstrap?.curriculum || { stages: [] });
  const fallbackWeight = (stage) => {
    const positive = Object.values(stage?.weights || {}).map((value) => Number(value)).filter((value) => Number.isFinite(value) && value > 0);
    return positive.length ? Math.min(...positive) : 0.05;
  };
  const stateDefinitions = {};
  enabledStates.forEach((state) => {
    const definition = clone(state.definition || {});
    if (state.kind === "replay_csv") {
      stateDefinitions[state.id] = { kind: "replay_csv", replay_folder: definition.replay_folder || "DataState", mirrored: Boolean(definition.mirrored) };
      return;
    }
    if (state.kind === "builtin") {
      stateDefinitions[state.id] = { kind: "builtin", base_id: definition.base_id || state.id, params: clone(definition.params || {}) };
      return;
    }
    stateDefinitions[state.id] = { kind: "scenario_inline", definition };
  });
  curriculum.state_definitions = stateDefinitions;
  curriculum.stages = (curriculum.stages || []).map((stage) => {
    const nextWeights = {};
    const missingWeight = fallbackWeight(stage);
    enabledStates.forEach((state) => {
      const explicit = Number(stage?.weights?.[state.id]);
      nextWeights[state.id] = Number.isFinite(explicit) && explicit > 0 ? explicit : missingWeight;
    });
    return { ...stage, weights: nextWeights };
  });
  return curriculum;
}

function metricSummary(row, preferredKeys) {
  if (!row) return "-";
  const preferred = preferredKeys.find((key) => row[key] != null && Number.isFinite(Number(row[key])));
  if (preferred) return `${preferred}: ${formatNumber(row[preferred])}`;
  const candidate = Object.entries(row).find(([, value]) => Number.isFinite(Number(value)));
  return candidate ? `${candidate[0]}: ${formatNumber(candidate[1])}` : "-";
}

function buildTrainingPayload() {
  const bundle = {
    config: clone(appState.launch.config),
    rewards: { weights: clone(selectedRewardDraft()) },
    curriculum: buildStateCurriculumOverride(),
  };
  bundle.config.environment.team_size = teamSizeFromMode();
  bundle.config.project.device = appState.launch.config.project.device;
  bundle.config.project.run_name_prefix = computedRunPrefix();
  bundle.config.training.opponent_mix = bundle.config.training.opponent_mix || {};
  bundle.config.training.opponent_mix.opponent_pool = clone(appState.launch.opponents || []).filter((entry) => Number(entry.weight) > 0).map((entry) => ({ name: entry.name, weight: Number(entry.weight) }));
  if (!bundle.config.training.opponent_mix.opponent_pool.length) bundle.config.training.opponent_mix.opponent_pool = [{ name: "self_play", weight: 1.0 }];
  return {
    title: `Training ${computedModelName()}`,
    display_name: computedModelName(),
    notes: appState.launch.notes || "",
    mode: selectedMode()?.name || "Duel",
    run_name_prefix: computedRunPrefix(),
    device: appState.launch.config.project.device,
    bundle,
  };
}

function buildReplayPayload() {
  const replayState = appState.launch.replays || {};
  const extraDirs = String(replayState.extra_replay_dirs || "").split(/[\n,]+/).map((item) => item.trim()).filter(Boolean);
  const replayDirs = unique([...(replayState.replay_dirs || []), ...extraDirs]);
  return {
    title: `Replay pretraining ${computedModelName()}`,
    display_name: computedModelName(),
    notes: appState.launch.notes || "",
    mode: selectedMode()?.name || "Duel",
    run_name_prefix: computedRunPrefix(),
    device: appState.launch.config.project.device,
    team_size: teamSizeFromMode(),
    replay_dirs: replayDirs,
    max_replays: replayState.use_all_replays ? null : toNumber(replayState.max_replays, 0),
    validation_replays: toNumber(replayState.validation_replays, 50),
    sample_fps: toNumber(replayState.sample_fps, 2),
    state_timeout_override: replayState.state_timeout_override === "" ? null : toNumber(replayState.state_timeout_override, 0),
    bundle: {
      config: {
        project: { device: appState.launch.config.project.device, run_name_prefix: computedRunPrefix() },
        environment: { team_size: teamSizeFromMode() },
      },
    },
  };
}

function renderModels() {
  const runs = appState.bootstrap?.runs || [];
  const cards = runs.map((run) => `
    <button type="button" class="run-card run-select ${run.run_name === appState.selectedRun ? "active" : ""}" data-run-name="${escapeAttr(run.run_name)}">
      <h4>${escapeHtml(run.display_name || run.run_name)}</h4>
      <p>${escapeHtml(run.run_name)}</p>
      <div class="tag-row"><span class="badge">${escapeHtml(run.mode || "custom")}</span><span class="badge">${escapeHtml(run.run_type || "run")}</span></div>
      <p style="margin-top:10px;">Train: ${escapeHtml(metricSummary(run.latest_training, ["mean_episode_return", "bc_loss", "bc_accuracy", "loss"]))}</p>
      <p>Eval: ${escapeHtml(metricSummary(run.latest_evaluation, ["win_rate", "goals_for_mean", "avg_goal_diff"]))}</p>
    </button>
  `).join("");
  if (!appState.runDetail) return `<div class="run-grid"><div class="card list-panel">${runs.length ? cards : `<div class="notice warning">Aucun run detecte.</div>`}</div><div class="detail-panel">${emptyCard("Selectionne un run pour voir ses courbes, checkpoints et actions d'evaluation/export.")}</div></div>`;
  const detail = appState.runDetail;
  const checkpoints = detail.checkpoints || [];
  const checkpointOptions = [`<option value="">dernier checkpoint</option>`].concat(checkpoints.map((item) => `<option value="${escapeAttr(item.name)}">${escapeHtml(item.name)}</option>`)).join("");
  return `
    <div class="run-grid">
      <div class="card list-panel">${runs.length ? cards : `<div class="notice warning">Aucun run detecte.</div>`}</div>
      <div class="detail-panel">
        <div class="card"><div class="section-header"><div><h3>${escapeHtml(detail.display_name || detail.run_name)}</h3><p>${escapeHtml(detail.run_name)}</p></div><div class="tag-row"><span class="badge">${escapeHtml(detail.meta?.mode || "custom")}</span><span class="badge">${escapeHtml(detail.meta?.run_type || "run")}</span></div></div><div class="kpi-grid"><div class="kpi"><span>Dernier training</span><strong>${escapeHtml(metricSummary(detail.training_metrics?.slice(-1)[0], ["mean_episode_return", "bc_loss", "bc_accuracy", "loss"]))}</strong></div><div class="kpi"><span>Derniere evaluation</span><strong>${escapeHtml(metricSummary(detail.evaluation_metrics?.slice(-1)[0], ["win_rate", "goals_for_mean", "avg_goal_diff"]))}</strong></div><div class="kpi"><span>Checkpoints</span><strong>${formatNumber(checkpoints.length, 0)}</strong></div></div></div>
        <div class="grid-2"><div class="card"><div class="section-header"><div><h3>Evaluation graphique</h3><p>Choix de l'adversaire, du checkpoint et du rendu 2D.</p></div></div><div class="form-grid"><div class="field"><label>Checkpoint</label><select id="eval-checkpoint">${checkpointOptions}</select></div><div class="field"><label>Adversaire</label><select id="eval-opponent">${(appState.bootstrap?.opponent_choices || []).map((choice) => `<option value="${escapeAttr(choice.id)}">${escapeHtml(choice.label)}</option>`).join("")}</select></div><div class="field"><label>Matchs</label><input id="eval-matches" type="number" step="1" value="16"></div><div class="field"><label>Render 2D</label><select id="eval-render"><option value="true" selected>true</option><option value="false">false</option></select></div></div><div class="inline" style="margin-top:14px;"><button id="run-evaluate-btn" class="primary-btn">Lancer l'evaluation</button></div></div><div class="card"><div class="section-header"><div><h3>Export RLBot</h3><p>Choix du checkpoint, du nom exporte et du chemin de sortie.</p></div></div><div class="form-grid"><div class="field"><label>Checkpoint</label><select id="export-checkpoint">${checkpointOptions}</select></div><div class="field"><label>Nom exporte</label><input id="export-display-name" type="text" value="${escapeAttr(detail.display_name || detail.run_name)}"></div><div class="field" style="grid-column:1 / -1;"><label>Output path</label><input id="export-output-path" type="text" value=""></div></div><div class="inline" style="margin-top:14px;"><button id="run-export-btn" class="primary-btn">Exporter pour RLBot</button></div></div></div>
        <div class="card"><div class="section-header"><div><h3>Courbes</h3><p>Progression de l'entrainement, evaluations et rewards.</p></div></div><div class="chart-wrap"><canvas id="training-chart" class="chart"></canvas><canvas id="evaluation-chart" class="chart"></canvas><canvas id="reward-chart" class="chart"></canvas></div></div>
      </div>
    </div>
  `;
}

function renderJobs() {
  const jobs = appState.bootstrap?.jobs || [];
  const cards = jobs.map((job) => `<button type="button" class="job-card job-select ${job.job_id === appState.selectedJob ? "active" : ""}" data-job-id="${escapeAttr(job.job_id)}"><h4>${escapeHtml(job.title || job.kind)}</h4><div class="tag-row"><span class="badge ${escapeAttr(job.status || "")}">${escapeHtml(job.status || "running")}</span><span class="badge">${escapeHtml(job.kind)}</span></div></button>`).join("");
  if (!appState.jobDetail) return `<div class="job-grid"><div class="card list-panel">${jobs.length ? cards : `<div class="notice warning">Aucun job actif dans cette session.</div>`}</div><div class="detail-panel">${emptyCard("Les jobs lances depuis l'interface apparaissent ici avec leur log en direct.")}</div></div>`;
  return `<div class="job-grid"><div class="card list-panel">${jobs.length ? cards : `<div class="notice warning">Aucun job actif dans cette session.</div>`}</div><div class="detail-panel"><div class="card"><div class="section-header"><div><h3>${escapeHtml(appState.jobDetail.title || appState.jobDetail.kind)}</h3><p>${escapeHtml(appState.jobDetail.job_id)}</p></div><div class="inline"><button id="job-refresh-log-btn" class="ghost-btn">Rafraichir le log</button></div></div><div class="kpi-grid"><div class="kpi"><span>Status</span><strong>${escapeHtml(appState.jobDetail.status || "-")}</strong></div><div class="kpi"><span>Type</span><strong>${escapeHtml(appState.jobDetail.kind || "-")}</strong></div><div class="kpi"><span>PID</span><strong>${escapeHtml(appState.jobDetail.pid ?? "-")}</strong></div><div class="kpi"><span>Retour</span><strong>${escapeHtml(appState.jobDetail.return_code ?? "-")}</strong></div></div></div><div class="card"><div class="section-header"><div><h3>Log temps reel</h3><p>Vue condensee du job en cours ou termine.</p></div></div><pre class="log-box">${escapeHtml(appState.jobLog || "Aucun log disponible.")}</pre></div></div></div>`;
}

function bindLaunchEvents() {
  document.querySelectorAll(".mode-select").forEach((button) => button.addEventListener("click", () => {
    appState.launch.modeId = button.dataset.modeId;
    renderAll();
  }));
  document.querySelectorAll("[data-launch-config]").forEach((input) => {
    const update = () => {
      const path = input.dataset.launchConfig;
      const doc = docByKey(path);
      let value = input.value;
      if (doc?.type === "bool") value = input.value === "true";
      else if (doc?.type === "int") value = toNumber(input.value, 0);
      else if (doc?.type === "float") value = toNumber(input.value, 0);
      setByPath(appState.launch.config, path, value);
    };
    input.addEventListener("change", update);
    input.addEventListener("input", update);
  });
  document.getElementById("launch-model-select")?.addEventListener("change", (event) => { appState.launch.selectedModelId = event.target.value; renderAll(); });
  document.getElementById("launch-new-model-name")?.addEventListener("input", (event) => { appState.launch.newModelName = event.target.value; });
  document.getElementById("launch-device-select")?.addEventListener("change", (event) => { appState.launch.config.project.device = event.target.value; });
  document.getElementById("launch-seed-input")?.addEventListener("input", (event) => { appState.launch.config.project.seed = toNumber(event.target.value, 0); });
  document.getElementById("launch-random-seed")?.addEventListener("click", () => {
    appState.launch.config.project.seed = Math.floor(Math.random() * 2147483647);
    const input = document.getElementById("launch-seed-input");
    if (input) input.value = String(appState.launch.config.project.seed);
  });
  document.getElementById("launch-notes")?.addEventListener("input", (event) => { appState.launch.notes = event.target.value; });
  document.querySelectorAll("[data-opponent-name]").forEach((input) => {
    const update = () => {
      const row = (appState.launch.opponents || []).find((entry) => entry.name === input.dataset.opponentName);
      if (!row) return;
      row.weight = Math.max(0, toNumber(input.value, 0));
    };
    input.addEventListener("change", update);
    input.addEventListener("input", update);
  });
  document.getElementById("launch-train-btn")?.addEventListener("click", async () => {
    try {
      const result = await postJSON("/api/jobs/train", buildTrainingPayload());
      window.alert(`Job lance: ${result.job_id}`);
      await refreshBootstrap(true);
    } catch (error) { window.alert(String(error.message || error)); }
  });
}

function bindRewardEvents() {
  document.getElementById("reward-profile-select")?.addEventListener("change", (event) => {
    if (event.target.value === "__new__") {
      appState.rewardEditor.profileMode = "new";
      appState.rewardEditor.newProfileName = "";
      appState.rewardEditor.draftWeights = clone(selectedRewardDraft());
    } else {
      const profile = rewardProfiles().find((item) => item.id === event.target.value) || rewardProfiles()[0];
      appState.rewardEditor.profileMode = "existing";
      appState.rewardEditor.selectedProfileId = profile?.id || "default";
      appState.rewardEditor.draftWeights = clone(profile?.weights || {});
    }
    renderAll();
  });
  document.getElementById("reward-new-profile-name")?.addEventListener("input", (event) => { appState.rewardEditor.newProfileName = event.target.value; });
  document.querySelectorAll("[data-reward-reward-enabled]").forEach((checkbox) => checkbox.addEventListener("change", () => {
    const rewardId = checkbox.dataset.rewardRewardEnabled;
    if (checkbox.checked) appState.rewardEditor.draftWeights[rewardId] = defaultRewardWeight(rewardId);
    else appState.rewardEditor.draftWeights[rewardId] = 0;
    renderAll();
  }));
  document.querySelectorAll("[data-reward-reward-weight]").forEach((input) => {
    const update = () => { appState.rewardEditor.draftWeights[input.dataset.rewardRewardWeight] = toNumber(input.value, 0); };
    input.addEventListener("change", update);
    input.addEventListener("input", update);
  });
  document.getElementById("save-reward-profile-btn")?.addEventListener("click", async () => {
    try {
      const payload = appState.rewardEditor.profileMode === "new"
        ? { name: appState.rewardEditor.newProfileName || "Nouveau profil", weights: clone(appState.rewardEditor.draftWeights), set_active: true }
        : { id: appState.rewardEditor.selectedProfileId, name: selectedRewardProfileName(), weights: clone(appState.rewardEditor.draftWeights), set_active: true };
      await postJSON("/api/reward-profiles/save", payload);
      await refreshBootstrap(true);
    } catch (error) { window.alert(String(error.message || error)); }
  });
}

function bindReplayEvents() {
  const replayState = appState.launch.replays;
  const bind = (id, key, parser = (value) => value, rerender = false) => {
    const input = document.getElementById(id);
    if (!input) return;
    const update = () => {
      replayState[key] = parser(input.value);
      if (rerender) renderAll();
    };
    input.addEventListener("change", update);
    input.addEventListener("input", update);
  };
  bind("replay-use-all", "use_all_replays", (value) => value === "true", true);
  bind("replay-max-replays", "max_replays", (value) => toNumber(value, 0));
  bind("replay-validation", "validation_replays", (value) => toNumber(value, 50));
  bind("replay-sample-fps", "sample_fps", (value) => toNumber(value, 2));
  bind("replay-timeout", "state_timeout_override");
  bind("replay-extra-dirs", "extra_replay_dirs");
  document.querySelectorAll("[data-replay-path]").forEach((checkbox) => checkbox.addEventListener("change", () => {
    const path = checkbox.dataset.replayPath;
    if (checkbox.checked) replayState.replay_dirs = unique([...(replayState.replay_dirs || []), path]);
    else replayState.replay_dirs = (replayState.replay_dirs || []).filter((item) => item !== path);
  }));
  document.getElementById("launch-pretrain-btn")?.addEventListener("click", async () => {
    try {
      const result = await postJSON("/api/jobs/pretrain", buildReplayPayload());
      window.alert(`Pretraining lance: ${result.job_id}`);
      await refreshBootstrap(true);
    } catch (error) { window.alert(String(error.message || error)); }
  });
}

function bindStateEvents() {
  document.querySelectorAll(".state-row").forEach((row) => row.addEventListener("click", async (event) => {
    if (event.target.matches("input")) return;
    await loadStateDetail(row.dataset.stateId);
  }));
  document.querySelectorAll("[data-state-toggle]").forEach((checkbox) => checkbox.addEventListener("change", async () => {
    const state = clone((appState.bootstrap?.state_catalog || []).find((item) => item.id === checkbox.dataset.stateToggle));
    if (!state) return;
    state.enabled = checkbox.checked;
    await postJSON("/api/states/save", { id: state.id, state });
    await refreshBootstrap(true);
  }));
  if (appState.stateDetail) {
    const detail = appState.stateDetail;
    ensureScenarioDefinition(detail);
    document.getElementById("toggle-state-preview-size")?.addEventListener("click", () => {
      appState.statePreviewExpanded = !appState.statePreviewExpanded;
      renderAll();
    });
    const previewCanvas = document.getElementById("state-detail-preview");
    if (previewCanvas && detail.kind !== "replay_csv") {
      drawTopDownPreview(previewCanvas, detail);
      previewCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
      let dragging = false;
      const finishDrag = (status = "Pret") => {
        if (!dragging) return;
        dragging = false;
        appState.stateEditor = { ...(appState.stateEditor || { selectedKind: "ball", selectedIndex: 0 }), dragMode: null, status };
        renderAll();
      };
      previewCanvas.addEventListener("mousedown", (event) => {
        const rect = previewCanvas.getBoundingClientRect();
        const picked = pickPreviewEntity(previewCanvas, detail, event.clientX - rect.left, event.clientY - rect.top);
        if (!picked) return;
        const changed = appState.stateEditor?.selectedKind !== picked.kind || appState.stateEditor?.selectedIndex !== picked.index;
        const alt = event.altKey;
        const ctrl = event.ctrlKey || event.metaKey;
        let dragMode = "position";
        let status = "Deplacement";
        if (event.button === 2) {
          dragMode = alt && picked.kind === "car" ? "yaw_range" : "random_box";
          status = dragMode === "yaw_range" ? "Edition de la plage d'angle" : "Edition de la zone random";
        } else if (alt && picked.kind === "car") {
          dragMode = "yaw";
          status = "Edition de l'angle";
        } else if (ctrl) {
          dragMode = "velocity";
          status = "Edition de la vitesse";
        }
        appState.stateEditor = { selectedKind: picked.kind, selectedIndex: picked.index, dragMode, status };
        if (changed) {
          renderAll();
          return;
        }
        dragging = true;
      });
      ["mouseup", "mouseleave"].forEach((name) => previewCanvas.addEventListener(name, () => finishDrag()));
      previewCanvas.addEventListener("mousemove", (event) => {
        if (!dragging) return;
        const entity = selectedStateEntity();
        if (!entity) return;
        const rect = previewCanvas.getBoundingClientRect();
        const world = previewWorld(previewCanvas, event.clientX - rect.left, event.clientY - rect.top);
        const dragMode = appState.stateEditor?.dragMode || "position";
        if (dragMode === "position") {
          entity.target.position.x = world.x;
          entity.target.position.y = world.y;
          if (entity.target.position_random?.enabled) {
            const width = entity.target.position_random.max_x - entity.target.position_random.min_x;
            const height = entity.target.position_random.max_y - entity.target.position_random.min_y;
            entity.target.position_random.min_x = world.x - width / 2;
            entity.target.position_random.max_x = world.x + width / 2;
            entity.target.position_random.min_y = world.y - height / 2;
            entity.target.position_random.max_y = world.y + height / 2;
          }
        } else if (dragMode === "velocity") {
          entity.target.velocity.x = (world.x - entity.target.position.x) / 0.08;
          entity.target.velocity.y = (world.y - entity.target.position.y) / 0.08;
        } else if (dragMode === "yaw" && entity.kind === "car") {
          entity.target.yaw.value = Math.atan2(-(world.y - entity.target.position.y), world.x - entity.target.position.x);
        } else if (dragMode === "random_box") {
          const range = entity.target.position_random;
          range.enabled = true;
          range.min_x = Math.min(entity.target.position.x, world.x);
          range.max_x = Math.max(entity.target.position.x, world.x);
          range.min_y = Math.min(entity.target.position.y, world.y);
          range.max_y = Math.max(entity.target.position.y, world.y);
          range.min_z = entity.target.position.z;
          range.max_z = entity.target.position.z;
        } else if (dragMode === "yaw_range" && entity.kind === "car") {
          entity.target.yaw.enabled = true;
          const yaw = Math.atan2(-(world.y - entity.target.position.y), world.x - entity.target.position.x);
          entity.target.yaw.min = Math.min(entity.target.yaw.value, yaw);
          entity.target.yaw.max = Math.max(entity.target.yaw.value, yaw);
        }
        drawTopDownPreview(previewCanvas, detail);
      });
    }
    document.getElementById("state-name-input")?.addEventListener("input", (event) => { appState.stateDetail.name = event.target.value; });
    document.getElementById("state-category-input")?.addEventListener("input", (event) => { appState.stateDetail.category = event.target.value; });
    document.getElementById("state-description-input")?.addEventListener("input", (event) => { appState.stateDetail.description = event.target.value; });
    document.getElementById("state-reward-profile-select")?.addEventListener("change", (event) => { appState.stateDetail.reward_profile = event.target.value; });
    document.getElementById("state-supported-input")?.addEventListener("input", (event) => {
      appState.stateDetail.supported_team_sizes = event.target.value.split(",").map((value) => toNumber(value.trim(), 0)).filter((value) => value > 0);
    });
    document.querySelectorAll(".state-entity-select").forEach((button) => button.addEventListener("click", () => {
      appState.stateEditor = { selectedKind: button.dataset.stateSelectKind, selectedIndex: toNumber(button.dataset.stateSelectIndex, 0), dragMode: null, status: "Pret" };
      renderAll();
    }));
    document.querySelectorAll("[data-state-team]").forEach((button) => button.addEventListener("click", () => {
      setByPath(appState.stateDetail, button.dataset.stateTeam, toNumber(button.dataset.stateTeamValue, 0));
      ensureScenarioDefinition(appState.stateDetail);
      renderAll();
    }));
    document.getElementById("state-add-blue-car")?.addEventListener("click", () => {
      const scenario = ensureScenarioDefinition(appState.stateDetail);
      const blueCount = scenario.cars.filter((car) => Number(car.team) === 0).length;
      scenario.cars.push(defaultCarSpec(0, blueCount));
      appState.stateEditor = { selectedKind: "car", selectedIndex: scenario.cars.length - 1, dragMode: null, status: "Voiture bleue ajoutee" };
      renderAll();
    });
    document.getElementById("state-add-orange-car")?.addEventListener("click", () => {
      const scenario = ensureScenarioDefinition(appState.stateDetail);
      const orangeCount = scenario.cars.filter((car) => Number(car.team) === 1).length;
      scenario.cars.push(defaultCarSpec(1, orangeCount));
      appState.stateEditor = { selectedKind: "car", selectedIndex: scenario.cars.length - 1, dragMode: null, status: "Voiture orange ajoutee" };
      renderAll();
    });
    document.getElementById("state-delete-car")?.addEventListener("click", () => {
      if (appState.stateEditor?.selectedKind !== "car") return;
      const scenario = ensureScenarioDefinition(appState.stateDetail);
      if (scenario.cars.length <= 1) return;
      scenario.cars.splice(appState.stateEditor.selectedIndex, 1);
      appState.stateEditor = { selectedKind: "ball", selectedIndex: 0, dragMode: null, status: "Voiture supprimee" };
      renderAll();
    });
    document.querySelectorAll("[data-state-path]").forEach((input) => {
      const eventName = input.type === "checkbox" || input.tagName === "SELECT" ? "change" : "input";
      input.addEventListener(eventName, () => {
        const path = input.dataset.statePath;
        let value;
        if (input.dataset.stateType === "bool") value = input.checked;
        else if (input.dataset.stateType === "bool-string") value = input.value === "true";
        else if (input.dataset.stateType === "int") value = toNumber(input.value, 0);
        else if (input.dataset.stateType === "text") value = input.value;
        else value = toNumber(input.value, 0);
        setByPath(appState.stateDetail, path, value);
        ensureScenarioDefinition(appState.stateDetail);
        if (previewCanvas && detail.kind !== "replay_csv") drawTopDownPreview(previewCanvas, detail);
        if (input.dataset.stateType === "bool" || input.dataset.stateType === "int" || input.tagName === "SELECT") renderAll();
      });
      if (input.dataset.stateType === "text") {
        input.addEventListener("change", () => renderAll());
      }
    });
    document.getElementById("state-mirror-btn")?.addEventListener("click", async () => {
      try {
        const result = await postJSON("/api/scenarios/mirror", { scenario: clone(appState.stateDetail.definition || {}) });
        appState.stateDetail.definition = result.scenario;
        renderAll();
      } catch (error) { window.alert(String(error.message || error)); }
    });
    document.getElementById("save-state-btn")?.addEventListener("click", async () => {
      try {
        await postJSON("/api/states/save", { id: appState.stateDetail.id, state: clone(appState.stateDetail) });
        await refreshBootstrap(true);
      } catch (error) { window.alert(String(error.message || error)); }
    });
  }
}

function bindModelEvents() {
  document.querySelectorAll(".run-select").forEach((button) => button.addEventListener("click", async () => { await loadRunDetail(button.dataset.runName); }));
  if (appState.runDetail) {
    drawLineChart(document.getElementById("training-chart"), appState.runDetail.training_metrics || [], pickSeries(appState.runDetail.training_metrics || [], ["mean_episode_return", "bc_loss", "bc_accuracy", "loss"]));
    drawLineChart(document.getElementById("evaluation-chart"), appState.runDetail.evaluation_metrics || [], pickSeries(appState.runDetail.evaluation_metrics || [], ["win_rate", "goals_for_mean", "avg_goal_diff"]));
    drawLineChart(document.getElementById("reward-chart"), appState.runDetail.reward_components || [], pickSeries(appState.runDetail.reward_components || [], ["goal_reward", "ball_goal_progress", "touch_reward", "save_reward", "align_ball_goal"]));
    document.getElementById("run-evaluate-btn")?.addEventListener("click", async () => {
      try {
        const result = await postJSON("/api/jobs/evaluate", { title: `Evaluation ${appState.selectedRun}`, run_name: appState.selectedRun, checkpoint_name: document.getElementById("eval-checkpoint")?.value || null, opponent: document.getElementById("eval-opponent")?.value || null, matches: toNumber(document.getElementById("eval-matches")?.value, 16), render_2d: (document.getElementById("eval-render")?.value || "true") === "true" });
        window.alert(`Evaluation lancee: ${result.job_id}`);
        await refreshBootstrap(true);
      } catch (error) { window.alert(String(error.message || error)); }
    });
    document.getElementById("run-export-btn")?.addEventListener("click", async () => {
      try {
        const result = await postJSON("/api/jobs/export", { title: `Export ${appState.selectedRun}`, run_name: appState.selectedRun, checkpoint_name: document.getElementById("export-checkpoint")?.value || null, display_name: document.getElementById("export-display-name")?.value || appState.selectedRun, output_path: document.getElementById("export-output-path")?.value || null });
        window.alert(`Export lance: ${result.job_id}`);
        await refreshBootstrap(true);
      } catch (error) { window.alert(String(error.message || error)); }
    });
  }
}

function bindJobEvents() {
  document.querySelectorAll(".job-select").forEach((button) => button.addEventListener("click", async () => { await loadJobDetail(button.dataset.jobId); }));
  document.getElementById("job-refresh-log-btn")?.addEventListener("click", async () => { await loadJobDetail(appState.selectedJob); });
}

async function loadStateDetail(stateId, rerender = true) {
  if (!stateId) return;
  const sameState = appState.selectedStateId === stateId;
  const previousEditor = sameState && appState.stateEditor ? clone(appState.stateEditor) : null;
  appState.selectedStateId = stateId;
  appState.stateDetail = await fetchJSON(`/api/states/${encodeURIComponent(stateId)}`);
  appState.stateEditor = previousEditor || { selectedKind: "ball", selectedIndex: 0, dragMode: null, status: "Pret" };
  ensureScenarioDefinition(appState.stateDetail);
  if (rerender) renderAll();
}

async function loadRunDetail(runName, rerender = true) {
  if (!runName) return;
  appState.selectedRun = runName;
  appState.runDetail = await fetchJSON(`/api/runs/${encodeURIComponent(runName)}`);
  if (rerender) renderAll();
}

async function loadJobDetail(jobId, rerender = true) {
  if (!jobId) return;
  appState.selectedJob = jobId;
  const [detail, log] = await Promise.all([fetchJSON(`/api/jobs/${encodeURIComponent(jobId)}`), fetchJSON(`/api/jobs/${encodeURIComponent(jobId)}/log`)]);
  appState.jobDetail = detail;
  appState.jobLog = log.log || "";
  if (rerender) renderAll();
}

function renderSidebarMeta() {
  const target = document.getElementById("sidebar-meta");
  if (!target || !appState.bootstrap) return;
  target.innerHTML = `<div><strong>${formatNumber((appState.bootstrap.state_catalog || []).length, 0)}</strong> states JSON</div><div><strong>${formatNumber((appState.bootstrap.reward_profiles?.profiles || []).length, 0)}</strong> profils de rewards</div><div><strong>${formatNumber((appState.bootstrap.runs || []).length, 0)}</strong> runs detectes</div><div><strong>${formatNumber((appState.bootstrap.jobs || []).length, 0)}</strong> job(s) de session</div>`;
}

function renderAll() {
  const scrollState = captureScrollState();
  document.querySelectorAll(".nav-btn").forEach((button) => button.classList.toggle("active", button.dataset.view === appState.currentView));
  document.querySelectorAll(".view").forEach((section) => section.classList.remove("active"));
  document.getElementById(`view-${appState.currentView}`)?.classList.add("active");
  document.getElementById("view-title").textContent = VIEW_TITLES[appState.currentView] || "Dashboard";
  document.getElementById("view-launch").innerHTML = renderLaunch();
  document.getElementById("view-states").innerHTML = renderStates();
  document.getElementById("view-rewards").innerHTML = renderRewards();
  document.getElementById("view-replays").innerHTML = renderReplays();
  document.getElementById("view-models").innerHTML = renderModels();
  document.getElementById("view-jobs").innerHTML = renderJobs();
  renderSidebarMeta();
  bindLaunchEvents();
  bindRewardEvents();
  bindReplayEvents();
  bindStateEvents();
  bindModelEvents();
  bindJobEvents();
  restoreScrollState(scrollState);
}

async function refreshBootstrap(rerender = true) {
  if (appState.refreshing) return;
  appState.refreshing = true;
  try {
    const bootstrap = await fetchJSON("/api/bootstrap");
    appState.bootstrap = bootstrap;
    appState.docIndex = buildDocIndex(bootstrap.parameter_docs || {});
    if (!appState.launch || !appState.rewardEditor) {
      seedClientState(bootstrap);
    } else {
      const profileList = bootstrap.reward_profiles?.profiles || [];
      const activeProfileId = bootstrap.reward_profiles?.active_profile || profileList[0]?.id || "default";
      const resolvedProfileId = appState.rewardEditor.profileMode === "existing" && profileList.some((item) => item.id === appState.rewardEditor.selectedProfileId)
        ? appState.rewardEditor.selectedProfileId
        : activeProfileId;
      const resolvedProfile = profileList.find((item) => item.id === resolvedProfileId) || profileList[0] || { weights: {} };
      appState.rewardEditor.profileMode = "existing";
      appState.rewardEditor.selectedProfileId = resolvedProfile.id || activeProfileId;
      appState.rewardEditor.draftWeights = clone(resolvedProfile.weights || {});
    }
    if (!appState.selectedStateId && (bootstrap.state_catalog || []).length) await loadStateDetail(bootstrap.state_catalog[0].id, false);
    else if (appState.selectedStateId) { try { await loadStateDetail(appState.selectedStateId, false); } catch (_error) {} }
    if (!appState.selectedRun && (bootstrap.runs || []).length) await loadRunDetail(bootstrap.runs[0].run_name, false);
    else if (appState.selectedRun) { try { await loadRunDetail(appState.selectedRun, false); } catch (_error) {} }
    if (!appState.selectedJob && (bootstrap.jobs || []).length) await loadJobDetail(bootstrap.jobs[0].job_id, false);
    else if (appState.selectedJob) { try { await loadJobDetail(appState.selectedJob, false); } catch (_error) {} }
    if (rerender) renderAll();
  } catch (error) {
    window.alert(String(error.message || error));
  } finally {
    appState.refreshing = false;
  }
}

function bindShellEvents() {
  document.querySelectorAll(".nav-btn").forEach((button) => button.addEventListener("click", () => { appState.currentView = button.dataset.view; renderAll(); }));
  document.getElementById("refresh-btn")?.addEventListener("click", async () => { await refreshBootstrap(true); });
}

async function boot() {
  bindShellEvents();
  await refreshBootstrap(true);
  window.setInterval(async () => { await refreshBootstrap(!isEditingField()); }, 5000);
}

window.addEventListener("load", boot);

