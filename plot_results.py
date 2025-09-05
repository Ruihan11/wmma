#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def read_csv_results(filename='benchmark_results.csv'):
    """Read benchmark results from CSV file"""
    if not os.path.exists(filename):
        print(f"CSV file {filename} not found. Please run the benchmark first.")
        return None
    
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def parse_csv_results(df):
    """Parse CSV data to extract performance data"""
    sizes = sorted(df['M'].unique())
    results = {
        'Opt0': [],  # cuBLAS
        'Opt1': [],  # Shared Memory
        'Opt2': [],  # Block Tiling
        'Opt3': []   # Optimized
    }
    
    for size in sizes:
        size_data = df[df['M'] == size]
        
        for kernel_name, opt_key in [('WMMA Opt0 cuBLAS', 'Opt0'), 
                                   ('WMMA Opt1 Shared Memory', 'Opt1'), 
                                   ('WMMA Opt2 Block Tiling', 'Opt2'),
                                   ('WMMA Opt3 Optimized', 'Opt3')]:
            kernel_data = size_data[size_data['Kernel'] == kernel_name]
            if not kernel_data.empty:
                results[opt_key].append(kernel_data['GFLOPS'].iloc[0])
            else:
                results[opt_key].append(0.0)
    
    return sizes, results

def plot_results(sizes, results):
    """Create and save optimization roadmap line plot"""
    plt.figure(figsize=(12, 8))
    
    # Order by optimization level
    colors = ['#000000', '#ff7f0e', '#2ca02c', '#d62728']
    kernels = ['Opt0', 'Opt1', 'Opt2', 'Opt3']
    labels = ['Opt0: cuBLAS', 'Opt1: Shared Mem', 'Opt2: Block Tiling', 'Opt3: Optimized']
    markers = ['o', '^', 's', 'D']
    
    num_sizes = len(sizes)
    
    for i, kernel in enumerate(kernels):
        if kernel in results and len(results[kernel]) > 0:
            # Ensure data arrays match sizes
            kernel_data = results[kernel][:num_sizes]  # Truncate if longer
            while len(kernel_data) < num_sizes:  # Pad with zeros if shorter
                kernel_data.append(0.0)
            
            # Filter out zero values for better plotting
            valid_data = [(s, d) for s, d in zip(sizes, kernel_data) if d > 0]
            if valid_data:
                plot_sizes, plot_data = zip(*valid_data)
                plt.plot(plot_sizes, plot_data, 
                        label=labels[i], 
                        color=colors[i], 
                        marker=markers[i],
                        linewidth=2,
                        markersize=8,
                        alpha=0.8)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Performance (GFLOPS)')
    plt.title('WMMA Kernel Optimization Roadmap - Performance Across All Sizes')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis ticks
    plt.xticks(sizes, [f'{s}³' for s in sizes])
    
    plt.tight_layout()
    plt.savefig('wmma_optimization_roadmap.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'wmma_optimization_roadmap.png'")
    # plt.show()

def main():
    print("Reading benchmark results from CSV...")
    
    df = read_csv_results()
    if df is None:
        print("Failed to read CSV results")
        return
    
    print("Parsing CSV data...")
    sizes, results = parse_csv_results(df)
    
    if not sizes:
        print("No results found in CSV data")
        return
    
    print(f"Found results for {len(sizes)} matrix sizes")
    kernels = ['Opt0', 'Opt1', 'Opt2', 'Opt3']
    for i, size in enumerate(sizes):
        size_results = []
        for kernel in kernels:
            if kernel in results and i < len(results[kernel]):
                size_results.append(f"{kernel}={results[kernel][i]:.1f}")
        print(f"  {size}³: {', '.join(size_results)} GFLOPS")
    
    print("Creating optimization roadmap plot...")
    plot_results(sizes, results)

if __name__ == "__main__":
    main()