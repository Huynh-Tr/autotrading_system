#!/usr/bin/env python3
"""
Generate Mermaid Flowcharts - Converts Mermaid diagrams to images
"""

import re
import os
import subprocess
import sys
import platform

def extract_mermaid_diagrams(markdown_file):
    """Extract Mermaid diagrams from markdown file"""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all Mermaid code blocks
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
    
    diagrams = []
    for i, diagram in enumerate(mermaid_blocks):
        diagram_name = f"diagram_{i+1}"
        diagrams.append({
            'name': diagram_name,
            'content': diagram.strip()
        })
    
    return diagrams

def create_mermaid_files(diagrams):
    """Create individual Mermaid files for each diagram"""
    files_created = []
    
    for diagram in diagrams:
        filename = f"docs/{diagram['name']}.mmd"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(diagram['content'])
        files_created.append(filename)
        print(f"✅ Created {filename}")
    
    return files_created

def find_mmdc_executable():
    """Find the mmdc executable path"""
    # Try different possible locations
    possible_paths = [
        'mmdc',  # If it's in PATH
        'mmdc.exe',  # Windows executable
        os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python', 'Python313', 'Scripts', 'mmdc.exe'),
        os.path.join(sys.prefix, 'Scripts', 'mmdc.exe'),
        os.path.join(sys.prefix, 'bin', 'mmdc'),
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Found mmdc at: {path}")
                return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    return None

def generate_images_from_mermaid(mermaid_files):
    """Generate images from Mermaid files using mermaid-cli"""
    mmdc_path = find_mmdc_executable()
    
    if not mmdc_path:
        print("❌ mermaid-cli not found. Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'mermaid-cli'], check=True)
            mmdc_path = find_mmdc_executable()
            if not mmdc_path:
                print("❌ Failed to find mmdc after installation")
                return False
        except subprocess.CalledProcessError:
            print("❌ Failed to install mermaid-cli")
            return False
    
    generated_images = []
    
    for mermaid_file in mermaid_files:
        output_file = mermaid_file.replace('.mmd', '.png')
        svg_file = mermaid_file.replace('.mmd', '.svg')
        
        try:
            # Generate PNG
            subprocess.run([
                mmdc_path, 
                '-i', mermaid_file, 
                '-o', output_file,
                '-b', 'white',
                '-w', '1200',
                '-H', '800'
            ], check=True)
            generated_images.append(output_file)
            print(f"✅ Generated {output_file}")
            
            # Generate SVG
            subprocess.run([
                mmdc_path, 
                '-i', mermaid_file, 
                '-o', svg_file,
                '-b', 'white'
            ], check=True)
            generated_images.append(svg_file)
            print(f"✅ Generated {svg_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to generate image from {mermaid_file}: {e}")
            # Try alternative approach using Python
            print("🔄 Trying alternative approach...")
            if generate_images_alternative(mermaid_file, output_file, svg_file):
                generated_images.extend([output_file, svg_file])
    
    return generated_images

def generate_images_alternative(mermaid_file, png_file, svg_file):
    """Alternative method to generate images using Python libraries"""
    try:
        # Try using graphviz as alternative
        import graphviz
        
        # Read the mermaid content
        with open(mermaid_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a simple dot representation (basic conversion)
        dot_content = convert_mermaid_to_dot(content)
        
        # Generate graph
        graph = graphviz.Source(dot_content)
        graph.render(png_file.replace('.png', ''), format='png', cleanup=True)
        graph.render(svg_file.replace('.svg', ''), format='svg', cleanup=True)
        
        print(f"✅ Generated images using alternative method")
        return True
        
    except ImportError:
        print("❌ graphviz not available. Install with: pip install graphviz")
        return False
    except Exception as e:
        print(f"❌ Alternative method failed: {e}")
        return False

def convert_mermaid_to_dot(mermaid_content):
    """Convert simple Mermaid content to DOT format"""
    # This is a basic conversion for simple graphs
    dot_content = "digraph G {\n"
    dot_content += "  rankdir=TB;\n"
    dot_content += "  node [shape=box, style=filled];\n"
    
    # Extract nodes and connections (basic parsing)
    lines = mermaid_content.split('\n')
    for line in lines:
        line = line.strip()
        if '-->' in line:
            parts = line.split('-->')
            if len(parts) == 2:
                from_node = parts[0].strip()
                to_node = parts[1].strip()
                dot_content += f'  "{from_node}" -> "{to_node}";\n'
        elif '[' in line and ']' in line:
            # Extract node name
            node_name = line.split('[')[0].strip()
            if node_name:
                dot_content += f'  "{node_name}" [label="{node_name}"];\n'
    
    dot_content += "}"
    return dot_content

def create_html_viewer(diagrams, images):
    """Create an HTML viewer for the flowcharts"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Auto Trading System Flowcharts</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .flowchart-section {
            margin-bottom: 40px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .flowchart-title {
            font-size: 18px;
            font-weight: bold;
            color: #444;
            margin-bottom: 15px;
        }
        .flowchart-image {
            text-align: center;
            margin: 15px 0;
        }
        .flowchart-image img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .download-links {
            margin-top: 10px;
            text-align: center;
        }
        .download-links a {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background-color: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 14px;
        }
        .download-links a:hover {
            background-color: #0056b3;
        }
        .description {
            margin-top: 15px;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 5px;
            font-size: 14px;
            line-height: 1.5;
        }
        .mermaid-content {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({ startOnLoad: true });
    </script>
</head>
<body>
    <div class="container">
        <h1>🚀 Auto Trading System Flowcharts</h1>
        
        <div class="flowchart-section">
            <div class="flowchart-title">📊 System Architecture Flowchart</div>
            <div class="mermaid-content">
graph TB
    %% Data Layer
    subgraph "Data Layer"
        MD[Market Data<br/>Yahoo Finance, VNStock]
        HC[Historical Data Cache]
        RT[Real-time Price Feeds]
        CF[Configuration Files]
    end
    
    %% Core Layer
    subgraph "Core Layer"
        DM[Data Manager<br/>Data fetching, caching, validation]
        TE[Trading Engine<br/>Order execution, portfolio management]
        CM[Configuration Manager<br/>Settings, parameters]
        LM[Logging & Monitoring<br/>Performance tracking]
    end
    
    %% Strategy Layer
    subgraph "Strategy Layer"
        SMA[SMA Crossover Strategy]
        RSI[RSI Strategy]
        MACD[MACD Strategy]
        CS[Custom Strategies]
    end
    
    %% Technical Indicators
    subgraph "Technical Indicators"
        SMA_I[SMA<br/>Simple Moving Average]
        RSI_I[RSI<br/>Relative Strength Index]
        MACD_I[MACD<br/>Moving Average Convergence]
        BB[Bollinger Bands]
    end
    
    %% Risk Management Layer
    subgraph "Risk Management Layer"
        PS[Position Sizing<br/>Risk-based allocation]
        SL[Stop Loss & Take Profit<br/>Automatic execution]
        PRL[Portfolio Risk Limits<br/>Maximum exposure]
        DM_R[Drawdown Monitoring<br/>Real-time tracking]
    end
    
    %% Backtesting Layer
    subgraph "Backtesting Layer"
        BE[Backtest Engine<br/>Multi-strategy testing]
        PM[Performance Metrics<br/>Sharpe, drawdown, win rate]
        TT[Trade Tracking<br/>Complete history, analysis]
        RI[Risk Integration<br/>Risk metrics in backtesting]
    end
    
    %% Optimization Layer
    subgraph "Optimization Layer"
        PG[Parameter Grid<br/>Strategy parameter combinations]
        SO[Strategy Optimizer<br/>Risk-based optimization]
        PP[Parallel Processing<br/>Efficient multi-core execution]
        RA[Results Analysis<br/>Top parameters, reports]
    end
    
    %% Dashboard Layer
    subgraph "Dashboard Layer"
        WD[Web Dashboard<br/>Streamlit-based monitoring]
        TC[Trading Charts<br/>Interactive OHLC charts]
        PR[Performance Reports<br/>Detailed analysis]
        RTM[Real-time Monitoring<br/>Live updates]
    end
    
    %% Output Layer
    subgraph "Output Layer"
        OS[Optimized Strategies<br/>Best parameters]
        BR[Backtest Results<br/>Performance data]
        RR[Risk Reports<br/>Risk metrics, alerts]
        TS[Trading Signals<br/>Buy/sell decisions]
    end
    
    %% Data Flow Connections
    MD --> DM
    HC --> DM
    RT --> TE
    CF --> CM
    
    DM --> SMA
    TE --> RSI
    CM --> MACD
    LM --> CS
    
    SMA --> SMA_I
    RSI --> RSI_I
    MACD --> MACD_I
    CS --> BB
    
    SMA --> PS
    RSI --> SL
    MACD --> PRL
    CS --> DM_R
    
    PS --> BE
    SL --> PM
    PRL --> TT
    DM_R --> RI
    
    SMA --> PG
    RSI --> SO
    MACD --> PP
    CS --> RA
    
    BE --> WD
    PM --> TC
    TT --> PR
    RI --> RTM
    
    PG --> PR
    SO --> RTM
    
    WD --> OS
    TC --> BR
    PR --> RR
    RTM --> TS
            </div>
            <div class="description">
                <strong>Description:</strong> This flowchart shows the complete system architecture with 9 layers: Data, Core, Strategy, Technical Indicators, Risk Management, Backtesting, Optimization, Dashboard, and Output. Each layer has specific components and data flows between them.
            </div>
        </div>
        
        <div class="flowchart-section">
            <div class="flowchart-title">🔄 Optimization Flowchart</div>
            <div class="mermaid-content">
graph TD
    %% Input Stage
    subgraph "Input Stage"
        ST[Strategy Type<br/>SMA, RSI, MACD]
        PG_I[Parameter Grid<br/>Predefined ranges]
        HD[Historical Data<br/>Price data]
        RP[Risk Parameters<br/>Stop loss, position sizing]
    end
    
    %% Process Stage
    subgraph "Process Stage"
        PC[Parameter Combinations<br/>Generate all combinations]
        SC[Strategy Creation<br/>Create strategy with parameters]
        BT[Backtesting<br/>Test strategy performance]
        RM[Risk Metrics<br/>Calculate risk metrics]
    end
    
    %% Optimization Stage
    subgraph "Optimization Stage"
        PE[Performance Evaluation<br/>Calculate metrics: Sharpe, return, drawdown]
        RANK[Ranking<br/>Sort by optimization metric]
        BP[Best Parameters<br/>Select top performing combination]
        VAL[Validation<br/>Verify results with different periods]
    end
    
    %% Output Stage
    subgraph "Output Stage"
        OS_O[Optimized Strategy<br/>Best parameters]
        PR_O[Performance Report<br/>Detailed analysis]
        PRANK[Parameter Rankings<br/>Top combinations]
        VIS[Visualization<br/>Charts and plots]
    end
    
    %% Flow Connections
    ST --> PC
    PG_I --> SC
    HD --> BT
    RP --> RM
    
    PC --> PE
    SC --> RANK
    BT --> BP
    RM --> VAL
    
    PE --> OS_O
    RANK --> PR_O
    BP --> PRANK
    VAL --> VIS
    
    %% Feedback Loop
    OS_O -.-> RP
            </div>
            <div class="description">
                <strong>Description:</strong> This flowchart illustrates the 4-stage optimization process: Input (strategy type, parameters), Process (combinations, creation, backtesting), Optimization (evaluation, ranking, best parameters), and Output (optimized strategy, reports). Includes a feedback loop for continuous improvement.
            </div>
        </div>
        
        <div class="flowchart-section">
            <div class="flowchart-title">📈 Data Flow Sequence Diagram</div>
            <div class="mermaid-content">
sequenceDiagram
    participant Market as Market Data Sources
    participant DataMgr as Data Manager
    participant Trading as Trading Engine
    participant Strategy as Strategy Layer
    participant Risk as Risk Manager
    participant Backtest as Backtest Engine
    participant Optimizer as Strategy Optimizer
    participant Dashboard as Dashboard
    participant Output as Output Layer
    
    Market->>DataMgr: Historical price data
    DataMgr->>Trading: Processed market data
    Trading->>Strategy: Market signals
    Strategy->>Risk: Trading signals
    Risk->>Backtest: Risk-adjusted signals
    Backtest->>Optimizer: Performance metrics
    Optimizer->>Strategy: Optimized parameters
    Strategy->>Trading: Enhanced signals
    Trading->>Dashboard: Real-time updates
    Dashboard->>Output: Trading decisions
    Output->>Market: Execute trades
            </div>
            <div class="description">
                <strong>Description:</strong> This sequence diagram shows the data flow between system components: Market Data Sources → Data Manager → Trading Engine → Strategy Layer → Risk Manager → Backtest Engine → Strategy Optimizer → Dashboard → Output Layer → Market (trade execution).
            </div>
        </div>
        
        <div class="description" style="margin-top: 30px; background-color: #d4edda; border: 1px solid #c3e6cb;">
            <strong>📝 Notes:</strong><br>
            • All flowcharts are generated from Mermaid diagrams in flowchart.md<br>
            • Interactive Mermaid diagrams are embedded in this HTML viewer<br>
            • The system architecture supports modular design and easy extension<br>
            • Risk management is integrated at every level of the system<br>
            • You can view the raw Mermaid code in the gray boxes above
        </div>
    </div>
</body>
</html>
"""
    
    with open('docs/flowcharts_viewer.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Created flowcharts_viewer.html")

def main():
    """Generate flowcharts from Mermaid diagrams"""
    print("🚀 Generating flowcharts from Mermaid diagrams...")
    
    markdown_file = 'docs/flowchart.md'
    
    if not os.path.exists(markdown_file):
        print(f"❌ Markdown file not found: {markdown_file}")
        return
    
    try:
        # Extract Mermaid diagrams
        print("📖 Extracting Mermaid diagrams from flowchart.md...")
        diagrams = extract_mermaid_diagrams(markdown_file)
        print(f"✅ Found {len(diagrams)} Mermaid diagrams")
        
        # Create individual Mermaid files
        print("📝 Creating individual Mermaid files...")
        mermaid_files = create_mermaid_files(diagrams)
        
        # Generate images
        print("🖼️ Generating images from Mermaid diagrams...")
        generated_images = generate_images_from_mermaid(mermaid_files)
        
        if generated_images:
            print(f"✅ Generated {len(generated_images)} images")
        else:
            print("⚠️ No images were generated, but HTML viewer created")
        
        # Create HTML viewer (always create this)
        print("🌐 Creating HTML viewer...")
        create_html_viewer(diagrams, generated_images)
        
        print("\n🎉 Flowchart generation completed successfully!")
        print("\n📁 Generated files:")
        for file in mermaid_files:
            print(f"   - {file}")
        if generated_images:
            for image in generated_images:
                print(f"   - {image}")
        print("   - docs/flowcharts_viewer.html")
        print("\n🌐 Open docs/flowcharts_viewer.html in your browser to view the flowcharts!")
        
    except Exception as e:
        print(f"❌ Error generating flowcharts: {e}")
        raise

if __name__ == "__main__":
    main() 