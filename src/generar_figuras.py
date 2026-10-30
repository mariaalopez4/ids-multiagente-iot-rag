import matplotlib.pyplot as plt
import numpy as np
import os

# Create images directory if it doesn't exist
os.makedirs("../latex/images", exist_ok=True)

# Set style for academic paper - premium styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10.5,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'figure.figsize': (7.5, 4.2),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.color': '#e2e8f0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.7
})

# Define a cohesive professional palette
COLOR_PRIMARY = '#1a365d'    # Deep Navy
COLOR_SECONDARY = '#2b5c8f'  # Steel Blue
COLOR_ACCENT = '#c53030'     # Muted Crimson
COLOR_HIGHLIGHT = '#2e7d32'  # Forest Green
COLOR_MUTED = '#718096'      # Slate Gray
COLOR_SPAN = '#feebc8'       # Soft Muted Amber

def plot_tradeoff():
    # Data
    modes = ['Downtime\n(Traditional)', 'Phase 1B\n(LLM Analyst)', 'Phase 1A\n(Binary Fallback)', 'Phase 3\n(Expert Mode)']
    f1_scores = [0.0, 0.533, 0.944, 0.925]  # Exact average F1-Weighted from tables
    latency = [0, 1605, 6.07, 4.25]  # in ms
    
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))

    # Bar chart for F1-Score
    ax1.set_xlabel('System State', fontweight='bold', labelpad=10, fontsize=13)
    ax1.set_ylabel('F1-Score / Protection Level', color=COLOR_PRIMARY, fontweight='bold', fontsize=13)
    bars = ax1.bar(modes, f1_scores, color=COLOR_SECONDARY, alpha=0.85, edgecolor=COLOR_PRIMARY, linewidth=1, width=0.45, label='F1-Score')
    ax1.tick_params(axis='y', labelcolor=COLOR_PRIMARY, labelsize=12)
    ax1.tick_params(axis='x', labelsize=11.5)
    ax1.set_ylim(0, 1.1)
    
    # Grid configuration (only horizontal, clean)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')
    ax1.xaxis.grid(False)

    # Twin axis for latency
    ax2 = ax1.twinx()
    ax2.set_ylabel('Latency (ms) - Log Scale', color=COLOR_ACCENT, fontweight='bold', fontsize=13)
    
    x_pos = np.arange(len(modes))
    latency_plot = [np.nan, 1605, 6.07, 4.25]
    
    ax2.plot(x_pos, latency_plot, color=COLOR_ACCENT, marker='s', linestyle='-', linewidth=2, markersize=7, label='Latency')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=COLOR_ACCENT, labelsize=12)
    
    # Hide top spine for clean look
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.spines['left'].set_color(COLOR_PRIMARY)
    ax1.spines['bottom'].set_color(COLOR_MUTED)
    ax2.spines['right'].set_color(COLOR_ACCENT)

    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval*100:.1f}%', ha='center', va='bottom', color='#1e293b', fontsize=11.5, fontweight='semibold')
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, 0.02, '0%', ha='center', va='bottom', color='#1e293b', fontsize=11.5, fontweight='semibold')

    plt.title('Performance vs. Latency Trade-off Across Modes', fontweight='bold', pad=15, fontsize=14)
    fig.tight_layout()
    plt.savefig('../latex/images/tradeoff_bar_chart.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_recovery_timeline():
    # Data for Edge-IIoT
    time_sec = np.linspace(-10, 200, 500)
    
    # Binary Protection (Always 94.4% after t=0)
    binary_f1 = np.piecewise(time_sec, [time_sec < 0, time_sec >= 0], [0.867, 0.944])
    
    # Multiclass Protection
    multi_f1 = np.piecewise(time_sec, 
                            [time_sec < 0, (time_sec >= 0) & (time_sec < 135), time_sec >= 135], 
                            [0.867, 0.450, 0.867])

    plt.figure(figsize=(6.5, 3.8))
    ax = plt.gca()
    
    # Plotting lines with professional colors
    plt.plot(time_sec, binary_f1 * 100, label='Binary Fallback (Phase 1A)', color=COLOR_HIGHLIGHT, linewidth=2.2, linestyle='--')
    plt.plot(time_sec, multi_f1 * 100, label='Multiclass Expert (Phase 3)', color=COLOR_PRIMARY, linewidth=2.2)
    
    # Soft background fill for cold start window
    plt.axvspan(0, 135, color=COLOR_SPAN, alpha=0.2, label='Cold Start Window (~2.2 min)')
    
    # Vertical lines for events
    plt.axvline(x=0, color=COLOR_ACCENT, linestyle=':', alpha=0.8, linewidth=1.8)
    plt.axvline(x=135, color=COLOR_SECONDARY, linestyle=':', alpha=0.8, linewidth=1.8)
    
    # Event labels - horizontal and clean, placed near the top but below the binary line
    plt.text(2.0, 78, 'Concept Drift\n(Schema Change)', color=COLOR_ACCENT, fontsize=10, fontweight='bold', ha='left', va='center')
    plt.text(133.0, 78, 'Hot-Swap\n(Model Deployed)', color=COLOR_SECONDARY, fontsize=10, fontweight='bold', ha='right', va='center')

    plt.xlabel('Time (seconds)', fontweight='bold', labelpad=6, fontsize=12)
    plt.ylabel('F1-Score (%)', fontweight='bold', labelpad=6, fontsize=12)
    plt.title('System Recovery Timeline (Edge-IIoTset)', fontweight='bold', pad=12, fontsize=13.5)
    plt.xlim(-15, 180)
    plt.ylim(30, 105)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10.5)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_MUTED)
    ax.spines['bottom'].set_color(COLOR_MUTED)
    
    # Grid and legend positioning
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig('../latex/images/recovery_timeline.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating figures...")
    plot_tradeoff()
    plot_recovery_timeline()
    print("Figures saved to ../latex/images/")

