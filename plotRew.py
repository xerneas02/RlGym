import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def read_data_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {}
    group = {}
    for line in lines:
        line = line.strip()
        if not line:  
            continue
        if line.startswith('20'):  
            if group:
                for reward_name, value in group.items():
                    if reward_name in data:
                        data[reward_name].append(value)
                    else:
                        data[reward_name] = [value]
                group = {}
        else:
            try:
                parts = line.split(': ')
                reward_name = parts[0]
                reward_value = float(parts[1])
                group[reward_name] = reward_value
            except:
                pass

    data = {reward_name: values for reward_name, values in data.items() if any(value != 0 for value in values)}

    return data


def remove_outliers(values):
    if not values:
        return values
    threshold = 1.5
    median = sorted(values)[len(values) // 2]
    std_dev = (sum((x - median) ** 2 for x in values) / len(values)) ** 0.5
    cleaned_values = [x if median - threshold * std_dev < x < median + threshold * std_dev else median for x in values]
    return cleaned_values


def plot_rewards(data, remove_outliers_flag=False, smoothness_percentage=10):
    num_rewards = len(data)
    num_plots_per_window = 3  # Nombre de graphiques par fenêtre
    num_windows = (num_rewards + num_plots_per_window - 1) // num_plots_per_window
    for window_index in range(num_windows):
        start_index = window_index * num_plots_per_window
        end_index = min((window_index + 1) * num_plots_per_window, num_rewards)
        fig, axes = plt.subplots(end_index - start_index, 1, figsize=(10, 6*(end_index - start_index)))
        if (end_index - start_index) == 1:
            axes = [axes]  
        for ax, (reward_name, values) in zip(axes, list(data.items())[start_index:end_index]):
            if remove_outliers_flag:
                values = remove_outliers(values)
            num_points = len(values)
            smoothness = int(num_points * smoothness_percentage / 100)
            smoothed_values = []
            for i in range(num_points):
                start = max(0, i - smoothness // 2)
                end = min(num_points, i + smoothness // 2 + 1)
                smoothed_value = sum(values[start:end]) / (end - start)
                smoothed_values.append(smoothed_value)
            x_values = range(1, num_points + 1) 
            ax.plot(x_values, values, label=reward_name, marker='o', linestyle='-')
            
            ax.plot(x_values, smoothed_values, color='red', linestyle='-', linewidth=3, label=f'{reward_name} (lissée)')
            
            ax.set_title(f'Récompense: {reward_name}')
            ax.set_xlabel('Simulation')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True)

        plt.tight_layout() 
        plt.show()

file_path = 'log_rew.txt'  
data = read_data_from_file(file_path)
plot_rewards(data, remove_outliers_flag=True, smoothness_percentage=10)
