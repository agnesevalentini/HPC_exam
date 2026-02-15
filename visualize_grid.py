#!/usr/bin/env python3
"""
Visualization script for heat diffusion stencil simulation
Reads binary snapshot files and creates visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import glob
import os
import struct

def read_grid_snapshot(filename):
    """Read a binary grid snapshot file"""
    with open(filename, 'rb') as f:
        # Read dimensions (2 unsigned ints)
        sizex = struct.unpack('I', f.read(4))[0]
        sizey = struct.unpack('I', f.read(4))[0]
        
        # Read grid data
        grid_data = np.fromfile(f, dtype=np.float64, count=sizex * sizey)
        grid = grid_data.reshape((sizey, sizex))
        
    return grid

def plot_single_snapshot(filename, save_png=True):
    """Plot a single grid snapshot"""
    grid = read_grid_snapshot(filename)
    
    # Extract iteration number from filename
    iter_num = int(filename.split('_')[-1].split('.')[0])
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(grid, cmap='hot', interpolation='bilinear', origin='lower')
    plt.colorbar(im, label='Energy/Temperature')
    plt.title(f'Heat Distribution - Iteration {iter_num}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    
    if save_png:
        png_name = filename.replace('.bin', '.png')
        plt.savefig(png_name, dpi=150)
        print(f"Saved: {png_name}")
    
    plt.show()

def plot_all_snapshots(pattern='grid_snapshot_*.bin', output_dir='snapshots'):
    """Plot all grid snapshots and save as PNG files"""
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(files)} snapshot files")
    
    for filename in files:
        grid = read_grid_snapshot(filename)
        iter_num = int(filename.split('_')[-1].split('.')[0])
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(grid, cmap='hot', interpolation='bilinear', origin='lower')
        plt.colorbar(im, label='Energy/Temperature')
        plt.title(f'Heat Distribution - Iteration {iter_num}')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'snapshot_{iter_num:04d}.png')
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved: {output_file}")

def create_animation(pattern='grid_snapshot_*.bin', output_file='heat_evolution.gif', 
                     fps=2, vmin=None, vmax=None):
    """Create an animated GIF showing the evolution over time"""
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Creating animation from {len(files)} snapshots...")
    
    # Read all grids
    grids = []
    iterations = []
    for filename in files:
        grid = read_grid_snapshot(filename)
        grids.append(grid)
        iter_num = int(filename.split('_')[-1].split('.')[0])
        iterations.append(iter_num)
    
    # Set color scale limits based on all data
    if vmin is None:
        vmin = min(grid.min() for grid in grids)
    if vmax is None:
        vmax = max(grid.max() for grid in grids)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(grids[0], cmap='hot', interpolation='bilinear', 
                   origin='lower', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='Energy/Temperature')
    title = ax.set_title(f'Heat Distribution - Iteration {iterations[0]}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    def update(frame):
        im.set_data(grids[frame])
        title.set_text(f'Heat Distribution - Iteration {iterations[frame]}')
        return [im, title]
    
    anim = FuncAnimation(fig, update, frames=len(grids), 
                        interval=1000//fps, blit=True, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)
    print(f"Animation saved: {output_file}")
    plt.close()

def compare_snapshots(filenames, titles=None):
    """Compare multiple snapshots side by side"""
    n = len(filenames)
    
    if n > 4:
        print("Warning: Showing only first 4 snapshots for comparison")
        filenames = filenames[:4]
        n = 4
    
    # Determine layout
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        rows, cols = 1, 2
    elif n <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3
    
    # Read grids
    grids = [read_grid_snapshot(f) for f in filenames]
    iterations = [int(f.split('_')[-1].split('.')[0]) for f in filenames]
    
    # Find global min/max for consistent color scale
    vmin = min(grid.min() for grid in grids)
    vmax = max(grid.max() for grid in grids)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (grid, iter_num) in enumerate(zip(grids, iterations)):
        if titles and i < len(titles):
            title = titles[i]
        else:
            title = f'Iteration {iter_num}'
        
        im = axes[i].imshow(grid, cmap='hot', interpolation='bilinear', 
                           origin='lower', vmin=vmin, vmax=vmax)
        axes[i].set_title(title)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        plt.colorbar(im, ax=axes[i], label='Energy')
    
    # Hide unused subplots
    for i in range(len(grids), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150)
    print("Saved: comparison.png")
    plt.show()

def analyze_evolution(pattern='grid_snapshot_*.bin'):
    """Analyze and plot statistics about the evolution"""
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    iterations = []
    total_energy = []
    max_temp = []
    mean_temp = []
    
    for filename in files:
        grid = read_grid_snapshot(filename)
        iter_num = int(filename.split('_')[-1].split('.')[0])
        
        iterations.append(iter_num)
        total_energy.append(grid.sum())
        max_temp.append(grid.max())
        mean_temp.append(grid.mean())
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    axes[0].plot(iterations, total_energy, 'b-o', markersize=4)
    axes[0].set_ylabel('Total Energy')
    axes[0].set_title('Energy Evolution Over Time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(iterations, max_temp, 'r-o', markersize=4)
    axes[1].set_ylabel('Maximum Temperature')
    axes[1].set_title('Peak Temperature Evolution')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(iterations, mean_temp, 'g-o', markersize=4)
    axes[2].set_ylabel('Mean Temperature')
    axes[2].set_xlabel('Iteration')
    axes[2].set_title('Average Temperature Evolution')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evolution_analysis.png', dpi=150)
    print("Saved: evolution_analysis.png")
    plt.show()

if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("Heat Diffusion Visualization Tool")
    print("=" * 60)
    
    # Check for snapshot files
    files = sorted(glob.glob('grid_snapshot_*.bin'))
    
    if not files:
        print("\nNo snapshot files found!")
        print("Make sure you have run the simulation first.")
        print("Files should be named: grid_snapshot_XXXX.bin")
        sys.exit(1)
    
    print(f"\nFound {len(files)} snapshot files")
    print(f"Iterations: {[int(f.split('_')[-1].split('.')[0]) for f in files[:5]]}{'...' if len(files) > 5 else ''}")
    
    print("\nWhat would you like to do?")
    print("1. Plot all snapshots as PNG images")
    print("2. Create animated GIF")
    print("3. Analyze evolution (statistics)")
    print("4. Compare specific snapshots")
    print("5. View a single snapshot")
    print("6. Do everything (1-3)")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        plot_all_snapshots()
    elif choice == '2':
        fps = input("Enter frames per second (default 2): ").strip()
        fps = int(fps) if fps else 2
        create_animation(fps=fps)
    elif choice == '3':
        analyze_evolution()
    elif choice == '4':
        print("\nAvailable snapshots:")
        for i, f in enumerate(files):
            iter_num = int(f.split('_')[-1].split('.')[0])
            print(f"  {i}: Iteration {iter_num}")
        indices = input("Enter indices to compare (e.g., 0 2 4): ").strip().split()
        selected = [files[int(i)] for i in indices if int(i) < len(files)]
        if selected:
            compare_snapshots(selected)
    elif choice == '5':
        print("\nAvailable snapshots:")
        for i, f in enumerate(files):
            iter_num = int(f.split('_')[-1].split('.')[0])
            print(f"  {i}: Iteration {iter_num}")
        idx = int(input("Enter index to view: ").strip())
        if 0 <= idx < len(files):
            plot_single_snapshot(files[idx])
    elif choice == '6':
        print("\n--- Plotting all snapshots ---")
        plot_all_snapshots()
        print("\n--- Creating animation ---")
        create_animation()
        print("\n--- Analyzing evolution ---")
        analyze_evolution()
    else:
        print("Invalid choice!")
    
    print("\nDone!")
