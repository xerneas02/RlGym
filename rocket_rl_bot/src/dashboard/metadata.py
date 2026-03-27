from __future__ import annotations

from pathlib import Path
from typing import Any

from src.dashboard.catalog import build_model_catalog, load_reward_profiles, load_state_library
from src.utils.config_loader import load_project_configs


PARAMETER_DOCS: dict[str, list[dict[str, Any]]] = {
    "project": [
        {
            "key": "project.run_name_prefix",
            "label": "Run name prefix",
            "type": "string",
            "description": "Prefixe technique utilise avant l'horodatage pour nommer les runs, logs et checkpoints.",
            "example": "ppo_goal_finish_v2",
            "advice": "Garde un prefixe stable par famille d'experiences pour comparer les runs plus facilement.",
        },
        {
            "key": "project.seed",
            "label": "Seed",
            "type": "int",
            "description": "Seed global pour Python, NumPy et PyTorch afin de rendre les experiences plus reproductibles.",
            "example": "42",
            "advice": "Conserve une seed fixe pour isoler un changement, puis varie-la pour confirmer un vrai gain.",
        },
        {
            "key": "project.device",
            "label": "Device",
            "type": "enum",
            "description": "Selectionne cpu, cuda ou auto pour l'entrainement et l'evaluation.",
            "example": "auto",
            "advice": "auto est le meilleur choix par defaut. Force cpu uniquement si ton GPU pose probleme.",
        },
    ],
    "environment": [
        {
            "key": "environment.num_envs",
            "label": "Parallel matches",
            "type": "int",
            "description": "Nombre de matchs simules en parallele pour collecter le rollout plus vite.",
            "example": "8",
            "advice": "Monte progressivement. Trop haut peut saturer le CPU et faire chuter le throughput reel.",
        },
        {
            "key": "environment.timeout_steps",
            "label": "Timeout steps",
            "type": "int",
            "description": "Duree maximale d'un episode en steps RL avant reset force.",
            "example": "450",
            "advice": "Augmente-le pour laisser plus de construction de jeu, baisse-le pour densifier les resets.",
        },
        {
            "key": "environment.no_touch_timeout_steps",
            "label": "No-touch timeout",
            "type": "int",
            "description": "Nombre de steps sans contact balle avant fin d'episode.",
            "example": "150",
            "advice": "Tres utile pour couper les phases mortes. Si ton bot hesite trop, cette valeur est souvent a reduire.",
        },
        {
            "key": "environment.end_on_goal",
            "label": "End on goal",
            "type": "bool",
            "description": "Si vrai, un but termine l'episode. Le dashboard force ensuite un reset en kickoff pour la relance suivante.",
            "example": "true",
            "advice": "Active-le pour apprendre des sequences courtes et lisibles.",
        },
        {
            "key": "environment.goal_reset_to_kickoff",
            "label": "Kickoff after goal",
            "type": "bool",
            "description": "Quand end_on_goal est vrai, le prochain reset est force sur un kickoff-like au lieu de repiocher dans le curriculum.",
            "example": "true",
            "advice": "Laisse cette option active pour garder une progression de match coherente.",
        },
    ],
    "training": [
        {
            "key": "training.total_steps",
            "label": "Total steps",
            "type": "int",
            "description": "Budget total d'entrainement PPO.",
            "example": "100000000",
            "advice": "Compare surtout la vitesse de progression sur les premiers millions de steps.",
        },
        {
            "key": "training.rollout_steps",
            "label": "Rollout steps",
            "type": "int",
            "description": "Longueur de collecte avant chaque update PPO.",
            "example": "8192",
            "advice": "Plus grand stabilise les stats mais retarde le feedback.",
        },
        {
            "key": "training.learning_rate",
            "label": "Learning rate",
            "type": "float",
            "description": "Taille du pas d'optimisation Adam pour la policy et la value.",
            "example": "0.0003",
            "advice": "Si la loss diverge ou si la policy oublie trop vite, baisse-la legerement.",
        },
        {
            "key": "training.entropy_coef",
            "label": "Entropy coef",
            "type": "float",
            "description": "Poids de l'exploration aleatoire dans l'objectif PPO.",
            "example": "0.01",
            "advice": "Plus haut au bootstrap, plus bas quand tu veux consolider un style de jeu.",
        },
        {
            "key": "training.checkpoint_interval_steps",
            "label": "Checkpoint interval",
            "type": "int",
            "description": "Frequence de sauvegarde des checkpoints PPO.",
            "example": "5000000",
            "advice": "Reduis cette valeur pendant les experiences risquees pour limiter la perte en cas d'instabilite.",
        },
    ],
    "evaluation": [
        {
            "key": "evaluation.num_matches",
            "label": "Evaluation matches",
            "type": "int",
            "description": "Nombre de matchs utilises pour estimer les performances d'un modele.",
            "example": "16",
            "advice": "Peu de matchs pour iterer vite, davantage pour comparer deux modeles serres.",
        },
        {
            "key": "evaluation.scripted_opponent",
            "label": "Default opponent",
            "type": "enum",
            "description": "Adversaire utilise par defaut pendant l'evaluation.",
            "example": "bronze_chaser",
            "advice": "Garde un adversaire fixe pour suivre une courbe propre.",
        },
        {
            "key": "evaluation.protocol",
            "label": "Evaluation protocol",
            "type": "string",
            "description": "Choisit si l'evaluation suit le curriculum courant ou un benchmark fige.",
            "example": "benchmark",
            "advice": "Le benchmark fixe est meilleur pour comparer deux runs.",
        },
    ],
    "pretraining": [
        {
            "key": "pretraining.max_replays",
            "label": "Replay limit",
            "type": "int",
            "description": "Nombre maximal de replays utilises pour le behavior cloning. 0 ou vide signifie tous.",
            "example": "2000",
            "advice": "Commence petit pour valider le parsing puis elargis le volume.",
        },
        {
            "key": "pretraining.validation_replays",
            "label": "Validation replays",
            "type": "int",
            "description": "Sous-ensemble reserve a la validation BC.",
            "example": "50",
            "advice": "Garde toujours un petit lot de validation pour verifier que tu n'overfit pas.",
        },
        {
            "key": "pretraining.sample_fps",
            "label": "Replay sample FPS",
            "type": "float",
            "description": "Cadence d'echantillonnage des frames replay converties en observations BC.",
            "example": "2.0",
            "advice": "Baisse ce FPS si le cache devient trop gros ou trop redondant.",
        },
        {
            "key": "pretraining.state_timeout_override",
            "label": "State timeout override",
            "type": "int",
            "description": "Override optionnel du timeout des states pendant le replay training.",
            "example": "220",
            "advice": "Utile pour un warmup plus nerveux sans reecrire la config globale.",
        },
    ],
}


REWARD_CATALOG: list[dict[str, Any]] = [
    {"id": "goal_reward", "name": "Goal Reward", "trainable": True, "description": "Objectif sparse principal. Recompense les buts marques et penalise les buts encaisses.", "example": "Poids eleve pour apprendre a prioriser le resultat du match.", "advice": "C'est generalement le poids directeur du systeme de reward."},
    {"id": "ball_goal_progress", "name": "Ball Goal Progress", "trainable": True, "description": "Shaping dense base sur la vraie progression de la balle vers le but adverse.", "example": "Augmenter ce poids favorise les poussees dangereuses.", "advice": "Tres utile au bootstrap offensif, mais evite de le rendre dominant face au score reel."},
    {"id": "velocity_to_ball", "name": "Velocity To Ball", "trainable": True, "description": "Recompense l'engagement vers la balle, surtout au kickoff et en challenge.", "example": "Un bot qui hesite devant la balle a souvent besoin d'un peu plus de ce terme.", "advice": "Monte-le si le bot flotte trop, baisse-le s'il sur-commit sans lecture."},
    {"id": "touch_reward", "name": "Touch Reward", "trainable": True, "description": "Recompense les contacts utiles qui creent vitesse ou progression.", "example": "Tres utile pour passer d'un bot qui court vers la balle a un bot qui la joue proprement.", "advice": "A coupler avec goal_reward et ball_goal_progress."},
    {"id": "defense_position", "name": "Defense Position", "trainable": True, "description": "Structure le placement: goal-side en defense, support derriere la balle en attaque.", "example": "Aide a reduire les over-commit et les retours trop lents.", "advice": "Garde-le leger pour eviter un bot trop passif."},
    {"id": "boost_efficiency", "name": "Boost Efficiency", "trainable": True, "description": "Petite pression contre le gaspillage de boost.", "example": "Peut lisser les lignes droites boostees inutiles.", "advice": "Laisse ce terme tres faible sauf si tu observes un vrai spam boost."},
    {"id": "ball_touch_reward", "name": "Legacy Ball Touch", "trainable": True, "description": "Reward legacy sur le simple contact balle, module par la hauteur.", "example": "Pratique pour densifier les premieres iterations.", "advice": "A garder plus faible que touch_reward pour eviter un bot qui cherche juste a toucher."},
    {"id": "distance_player_ball", "name": "Legacy Distance Player Ball", "trainable": True, "description": "Reward historique de proximite a la balle.", "example": "Accelere la convergence initiale quand le bot flotte loin de l'action.", "advice": "A garder leger sinon il peut sur-coller la balle sans lecture."},
    {"id": "align_ball_goal", "name": "Legacy Align Ball Goal", "trainable": True, "description": "Mesure un alignement geometrique entre voiture, balle et but.", "example": "Utile pour recompenser une ligne d'attaque propre.", "advice": "A utiliser comme shaping secondaire."},
    {"id": "save_reward", "name": "Legacy Save Reward", "trainable": True, "description": "Recompense les renversements de trajectoire apres un contact sauveur.", "example": "Specialement utile pour des curriculums de gardiennage.", "advice": "A monter seulement si tu veux vraiment densifier le comportement defensif."},
]


MODE_PRESETS: list[dict[str, Any]] = [
    {
        "id": "duel",
        "name": "Duel 1v1",
        "status": "ready",
        "description": "Preset le plus stable pour apprendre vite et evaluer proprement.",
        "config_overrides": {"config": {"environment": {"team_size": 1}, "project": {"run_name_prefix": "ppo_duel_dashboard"}}},
    },
    {
        "id": "2v2",
        "name": "2v2",
        "status": "experimental",
        "description": "Mode 2v2 avec la meme stack d'entrainement.",
        "config_overrides": {"config": {"environment": {"team_size": 2}, "project": {"run_name_prefix": "ppo_2v2_dashboard"}}},
    },
    {
        "id": "3v3",
        "name": "3v3",
        "status": "experimental",
        "description": "Mode 3v3. Plus lourd CPU et plus exploratoire avec l'observation compacte actuelle.",
        "config_overrides": {"config": {"environment": {"team_size": 3}, "project": {"run_name_prefix": "ppo_3v3_dashboard"}}},
    },
]


DEVICE_CHOICES = ["auto", "cuda", "cpu"]
OPPONENT_CHOICES = [
    {"id": "self_play", "label": "Self Play"},
    {"id": "historical_self_play", "label": "Historical Self Play"},
    {"id": "necto", "label": "Necto"},
    {"id": "seer", "label": "Seer"},
    {"id": "bronze_chaser", "label": "Bronze Chaser"},
    {"id": "aggressive_chaser", "label": "Aggressive Chaser"},
]


def build_dashboard_bootstrap(project_root: Path) -> dict[str, Any]:
    config, rewards, curriculum = load_project_configs(Path(project_root))
    reward_profiles = load_reward_profiles(Path(project_root))
    active_profile_id = str(reward_profiles.get("active_profile", "default"))
    active_profile = next((profile for profile in reward_profiles.get("profiles", []) if str(profile.get("id")) == active_profile_id), None)
    if active_profile:
        rewards["weights"] = dict(active_profile.get("weights", rewards.get("weights", {})))
    return {
        "config": config,
        "rewards": rewards,
        "curriculum": curriculum,
        "parameter_docs": PARAMETER_DOCS,
        "reward_catalog": REWARD_CATALOG,
        "state_catalog": load_state_library(Path(project_root)),
        "reward_profiles": reward_profiles,
        "mode_presets": MODE_PRESETS,
        "device_choices": DEVICE_CHOICES,
        "opponent_choices": OPPONENT_CHOICES,
        "model_catalog": build_model_catalog(Path(project_root)),
    }
