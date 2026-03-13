# Rocket RL Bot

## 1. Vision globale

`rocket_rl_bot` est un workspace RL 1v1 pour Rocket League, pense pour des entrainements longs sur machine personnelle avec une base simple a maintenir.

Choix structurants:
- environnement `rlgym-sim` oriente CPU, `tick_skip=8`, `frame_rate=120`, `num_envs=8`
- PPO PyTorch maison pour garder un pipeline lisible, controllable et facile a reprendre
- compatibilite `rocket-learn` conservee au niveau des dependances et du metadata run pour une migration distribuee future
- observation compacte de `126` features
- action space discret LUT de `96` actions
- reward system limite a `6` composantes, toutes loggees separement
- curriculum explicite avec paliers `0`, `10M`, `50M`, `100M` steps
- checkpoints complets avec poids, optimizer, seed, step et config
- analytics CSV + TensorBoard + scripts matplotlib/pandas

## 2. Arborescence

```text
rocket_rl_bot/
|-- analytics/
|   |-- analyze_rewards.py
|   |-- compare_runs.py
|   `-- plot_training_curves.py
|-- checkpoints/
|-- configs/
|   |-- curriculum.yaml
|   |-- rewards.yaml
|   `-- training.yaml
|-- logs/
|-- scripts/
|   |-- _bootstrap.py
|   |-- evaluate.py
|   |-- resume_training.py
|   `-- train.py
|-- src/
|   |-- env/
|   |   |-- actions.py
|   |   |-- env_builder.py
|   |   |-- observations.py
|   |   |-- rewards.py
|   |   |-- state_setters.py
|   |   `-- terminal_conditions.py
|   |-- rl/
|   |   |-- evaluator.py
|   |   |-- metrics.py
|   |   |-- model.py
|   |   `-- trainer.py
|   `-- utils/
|       |-- checkpointing.py
|       |-- logging_utils.py
|       `-- seeding.py
|-- README.md
`-- requirements.txt
```

## 3. Justification technique

### Environnement
- format cible: `1v1` self-play partage pendant le train
- execution: `8` matchs paralleles dans un `ThreadedVectorEnv`
- timeout match: `450` steps RL, soit `30` secondes simulees a `tick_skip=8`
- no-touch timeout: `150` steps RL, soit `10` secondes simulees

### Action space
Le LUT contient `96` actions.

Restrictions appliquees:
- pas de boost en marche arriere
- pas de `jump + handbrake`
- pas de roll inutile au sol
- pas de combinaisons aeriennes contradictoires trop rares pour apprendre proprement
- powerslide garde uniquement pour les virages serres

### Observation space
Observation finale: `126` features normalisees.

Contenu:
- balle: position, vitesse, vitesse angulaire
- voiture courante: position, vitesse, vitesse angulaire, forward, up, flags
- contexte relatif: balle relative, buts, murs
- pads: `34` features d'occupation
- adversaire: etat absolu + relatif
- action precedente: `8` features

### Reward system
Poids retenus:
- `goal_reward = 10.0`
- `ball_goal_progress = 0.02`
- `velocity_to_ball = 0.01`
- `touch_reward = 0.2`
- `defense_position = 0.005`
- `boost_efficiency = 0.001`

Interpretation:
- `goal_reward`: objectif dominant, sparse, non ambigu
- `ball_goal_progress`: densifie l'apprentissage offensif sans recompenser n'importe quel mouvement
- `velocity_to_ball`: reduit les hesitations et les trajectoires molles
- `touch_reward`: filtre les touches utiles au lieu de sur-recompenser tout contact
- `defense_position`: pousse le bot a rester goal-side quand la balle est dangereuse
- `boost_efficiency`: petite pression contre le gaspillage, sans sur-contraindre le style de jeu

### Curriculum
Distribution initiale:
- `kickoff_like 0.10`
- `open_goal_attack 0.25`
- `simple_defense 0.25`
- `ball_center_random 0.20`
- `wall_ball 0.10`
- `random_match_state 0.10`

Paliers:
- `10M`: moins de situations triviales, plus de variete
- `50M`: montee du wall play et du random state
- `100M`: forte dominante `random_match_state` pour consolider le jeu general

### Reseau PPO
- policy MLP: `256 -> 256 -> action logits`
- value MLP: `256 -> 256 -> scalar`
- activation: `ReLU`
- optimizer: `Adam`
- learning rate: `3e-4`

Hyperparametres:
- `gamma = 0.995`
- `gae_lambda = 0.95`
- `clip_range = 0.2`
- `entropy_coef = 0.01`
- `batch_size = 65536`
- `minibatch_size = 4096`
- `rollout_steps = 8192`
- `epochs = 3`
- `gradient clipping = 0.5`

## 4. Pipeline d'entrainement

Le trainer fait:
- collecte sur `8` environnements paralleles
- self-play symetrique pendant l'entrainement
- export TensorBoard
- export CSV
- evaluation toutes les `1M` steps
- checkpoint toutes les `2M` steps
- reprise depuis checkpoint explicitement ou automatiquement selon la config

Checkpoints sauvegardes:
- `model_state_dict`
- `optimizer_state_dict`
- `training_state`
- `config`
- `seed`

## 5. Evaluation

Commande:

```bash
python scripts/evaluate.py --checkpoint checkpoints/<run>/step_000002000000.pt
```

Evaluation par defaut:
- `20` matchs
- adversaire scripte `bronze_chaser`
- metriques:
  - `goal_rate`
  - `average_reward`
  - `touches_per_game`
  - `time_of_possession`
  - `concede_rate`
  - `average_speed`

## 6. Analytics

Scripts disponibles:
- `python analytics/plot_training_curves.py --run-dir logs/<run>`
- `python analytics/compare_runs.py --run-dirs logs/<run_a> logs/<run_b> --metric episode_reward`
- `python analytics/analyze_rewards.py --run-dir logs/<run>`

Graphiques couverts:
- reward curves
- goal rate
- touch rate
- training loss
- entropy
- composantes de reward

## 7. Installation et lancement

Depuis `rocket_rl_bot/`:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py
```

Pre-requis `rlgym-sim` a ne pas oublier:
- `rocketsim` doit etre installe dans le venv
- les assets Rocket League doivent etre dumpes avec `RLArenaCollisionDumper`
- les fichiers d'assets doivent etre places a la racine de `rocket_rl_bot/`

Reprise:

```bash
python scripts/resume_training.py
```

Reprise avec checkpoint precis:

```bash
python scripts/resume_training.py --checkpoint checkpoints/<run>/step_000010000000.pt
```

## 8. Preset recommande

Objectifs realistes avec ce preset `8 envs / PPO / 1v1 self-play`:
- `24h`: le bot touche la balle regulierement
- `72h`: le bot pousse la balle vers le but de facon consistante
- `7 jours`: le bot devient competitif contre un baseline bronze-like

## 9. Roadmap V2

Extensions recommandees:
- actions continues
- attention network
- multi-agent training plus riche
- self-play avec pool historique d'adversaires
- league training

## 10. Notes pratiques

- `simulation_speed_multiplier=100` est conserve dans la config pour compatibilite de preset. Sur `rlgym-sim`, la vitesse reelle depend surtout du throughput CPU et du nombre d'environnements.
- `rlgym-sim` 1.2.6 attend explicitement `RocketSim` et des assets arena dumpes localement. Sans ces assets, le simulateur ne pourra pas initialiser les matchs.
- le trainer est volontairement monolithique seulement sur la boucle PPO, mais les briques `env`, `rl`, `utils`, `analytics` restent separees pour modifier rewards, states ou logging sans casser le reste.
- le point d'entree principal est `scripts/train.py`.

## 11. Visualisation 2D simplifiee

Une vue 2D top-down est disponible cote evaluation pour observer:
- le terrain et les cages
- la balle
- les voitures
- les boosts
- les inputs de controle
- une indication de hauteur via labels `z` et jauge verticale

Evaluation avec viewer live:

```bash
python scripts/evaluate.py --checkpoint checkpoints/<run>/step_000002000000.pt --render-2d --fps 15
```

Evaluation avec export de trajectoire:

```bash
python scripts/evaluate.py --checkpoint checkpoints/<run>/step_000002000000.pt --save-trajectory logs/<run>/eval_topdown.json
```

Replay d'une trajectoire en 2D:

```bash
python analytics/replay_topdown.py --trajectory logs/<run>/eval_topdown.json --fps 15
```

Cette visualisation est volontairement separee de la boucle d'entrainement pour conserver un throughput maximal pendant les longues sessions.
