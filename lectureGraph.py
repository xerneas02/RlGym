import matplotlib.pyplot as plt
from collections import defaultdict

def lire_fichier(nom_fichier):
    data = defaultdict(list)

    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()
        for i in range(0, len(lignes), 5):  # Incrément de 5 pour chaque bloc de données
            simulation = int(lignes[i].strip())
            data[simulation].append(int(lignes[i+1].strip()))  # Buts marqués
            data[simulation].append(int(lignes[i+2].strip()))  # Balles touchées
            data[simulation].append(float(lignes[i+3].strip()))  # % Temps du bot

    simulations = []
    buts_marques_avg = []
    balles_touchees_avg = []
    pourcentage_temps_bot_avg = []

    for simulation, values in data.items():
        simulations.append(simulation)
        buts_marques_avg.append(sum(values[::3]) / len(values[::3]))
        balles_touchees_avg.append(sum(values[1::3]) / len(values[1::3]))
        pourcentage_temps_bot_avg.append(sum(values[2::3]) / len(values[2::3]))

    return simulations, buts_marques_avg, balles_touchees_avg, pourcentage_temps_bot_avg

def plot_courbe(nom_fichier):
    simulations, buts_marques, balles_touchees, pourcentage_temps_bot = lire_fichier(nom_fichier)

    plt.figure(figsize=(10, 6))

    plt.plot(simulations, buts_marques, marker='o', color='b', label='Buts marqués')
    plt.plot(simulations, balles_touchees, marker='o', color='r', label='Balles touchées')
    plt.plot(simulations, pourcentage_temps_bot, marker='o', color='g', label='% Temps du bot')
    plt.title('Comparaison des paramètres')
    plt.xlabel('Nombre de simulations')
    plt.ylabel('Valeurs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_statistics():
    # Fermer toutes les fenêtres existantes
    plt.close('all')

    # Lecture des données depuis le fichier stats_bot.txt
    with open("stats_bot.txt", "r") as file:
        lines = file.readlines()

    # Extraction des données en groupes de 5 lignes (4 paramètres + "ff")
    data_groups = [list(map(float, lines[i:i+4])) for i in range(0, len(lines), 5)]
    num_simulations = len(data_groups)

    # Dernier ensemble de données
    last_data = data_groups[-1]

    # Calcul de la moyenne des données
    num_groups = len(data_groups)
    avg_data = [sum(x) / num_groups for x in zip(*data_groups)]

    # Création des histogrammes
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Taille en pouces pour une résolution de 1920x1080

    # Histogramme pour les dernières valeurs
    labels = ['NUMBER_GOAL', 'NUMBER_TOUCH', 'BEHIND_BALL_TIME']
    bars = axs[0].bar(labels, last_data[1:], color='skyblue')

    # Affichage des valeurs sur les barres
    for bar, value in zip(bars, last_data[1:]):
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, height, round(value, 2), ha='center', va='bottom')

    axs[0].set_title('Dernières valeurs des paramètres')
    axs[0].set_ylabel('Valeur')

    # Histogramme pour les moyennes
    bars = axs[1].bar(labels, avg_data[1:], color='lightgreen')

    # Affichage des valeurs sur les barres
    for bar, value in zip(bars, avg_data[1:]):
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, height, round(value, 2), ha='center', va='bottom')

    axs[1].set_title('Moyenne des paramètres')
    axs[1].set_ylabel('Moyenne')

    # Affichage du nombre de simulations
    plt.figtext(0.5, 0.002, f"Nombre de simulations : {num_simulations}", ha='center', fontsize=12)

    # Affichage des graphiques sans bloquer l'exécution
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)  # Permet à la fenêtre de répondre

plot_statistics()
plot_courbe("stats_bot.txt")


