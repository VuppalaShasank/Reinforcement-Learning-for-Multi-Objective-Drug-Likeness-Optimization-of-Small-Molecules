import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def create_document(output_path):
    doc = docx.Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    # Styling Palette
    C_NAVY = RGBColor(27, 54, 93)      # #1B365D - Headers
    C_SLATE = RGBColor(74, 85, 104)    # #4A5568 - Secondary Text
    C_BLUE = RGBColor(37, 99, 235)     # #2563EB - Highlights & Links
    C_DARK = RGBColor(30, 41, 59)      # #1E293B - Body Text
    
    # Base Normal style
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Calibri'
    font.size = Pt(10)
    font.color.rgb = C_DARK

    def set_cell_background(cell, hex_color):
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    def set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = OxmlElement('w:tcMar')
        for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
            node = OxmlElement(f'w:{m}')
            node.set(qn('w:w'), str(val))
            node.set(qn('w:type'), 'dxa')
            tcMar.append(node)
        tcPr.append(tcMar)

    # 1. Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(3)
    run_title = p_title.add_run('Reinforcement Learning for Multi-Objective Drug-Likeness Optimization of Small Molecules')
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = C_NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(10)
    run_sub = p_sub.add_run('From Adaptive Policy Gradients to Policy-Guided Local Search')
    run_sub.font.size = Pt(11.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = C_SLATE

    # Horizontal Divider Line
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(8)
    p_div_border = parse_xml(r'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:bottom w:val="single" w:sz="12" w:space="1" w:color="1B365D"/></w:pBdr>')
    p_div._p.get_or_add_pPr().append(p_div_border)

    # 2. Metadata Table
    meta_table = doc.add_table(rows=6, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    meta_data = [
        ('Project Title', 'Reinforcement Learning for Multi-Objective Drug-Likeness Optimization of Small Molecules: From Adaptive Policy Gradients to Policy-Guided Local Search'),
        ('Student / Author', 'Shasank Vuppala (Roll No: U4AIE23064)'),
        ('Academic Department', 'Department of Computer Science and Engineering (Artificial Intelligence),\nAmrita Vishwa Vidyapeetham, Amritapuri Campus, Kerala, India'),
        ('Course Information', '22AIE401 — Reinforcement Learning (Semester 7 Capstone Project)'),
        ('Public GitHub Link', 'https://github.com/VuppalaShasank/Reinforcement-Learning-for-Multi-Objective-Drug-Likeness-Optimization-of-Small-Molecules'),
        ('Terminal Clone', 'git clone https://github.com/VuppalaShasank/Reinforcement-Learning-for-Multi-Objective-Drug-Likeness-Optimization-of-Small-Molecules.git')
    ]

    col_widths = [Inches(1.8), Inches(4.9)]
    for i, (label, val) in enumerate(meta_data):
        row = meta_table.rows[i]
        
        c0 = row.cells[0]
        c0.width = col_widths[0]
        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(2)
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(label)
        r0.font.bold = True
        r0.font.size = Pt(9)
        r0.font.color.rgb = C_NAVY
        set_cell_background(c0, 'F1F5F9')
        set_cell_margins(c0, 50, 50, 90, 90)
        
        c1 = row.cells[1]
        c1.width = col_widths[1]
        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(2)
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(val)
        r1.font.size = Pt(9)
        if 'http' in val or 'git clone' in val:
            r1.font.bold = True
            r1.font.color.rgb = C_BLUE
        set_cell_background(c1, 'FAFAFA' if i % 2 == 0 else 'FFFFFF')
        set_cell_margins(c1, 50, 50, 90, 90)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 3. Callout Box for Repository Link
    callout = doc.add_table(rows=1, cols=1)
    callout.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_cell = callout.rows[0].cells[0]
    c_cell.width = Inches(6.7)
    set_cell_background(c_cell, 'EFF6FF')
    set_cell_margins(c_cell, 100, 100, 150, 150)
    
    tcPr = c_cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:left w:val="single" w:sz="24" w:space="0" w:color="2563EB"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    tcPr.append(tcBorders)

    cp = c_cell.paragraphs[0]
    cp.paragraph_format.space_after = Pt(2)
    c_head = cp.add_run('🔗 Verified Public Source Code Repository:')
    c_head.font.bold = True
    c_head.font.size = Pt(10)
    c_head.font.color.rgb = C_NAVY

    cp2 = c_cell.add_paragraph()
    cp2.paragraph_format.space_after = Pt(2)
    c_url = cp2.add_run('https://github.com/VuppalaShasank/Reinforcement-Learning-for-Multi-Objective-Drug-Likeness-Optimization-of-Small-Molecules')
    c_url.font.bold = True
    c_url.font.size = Pt(9.5)
    c_url.font.color.rgb = C_BLUE

    cp3 = c_cell.add_paragraph()
    cp3.paragraph_format.space_after = Pt(0)
    c_clone = cp3.add_run('Command: git clone https://github.com/VuppalaShasank/Reinforcement-Learning-for-Multi-Objective-Drug-Likeness-Optimization-of-Small-Molecules.git')
    c_clone.font.size = Pt(8.5)
    c_clone.font.italic = True
    c_clone.font.color.rgb = C_SLATE

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    def add_section_heading(title):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(3)
        hrun = h.add_run(title)
        hrun.font.size = Pt(12)
        hrun.font.bold = True
        hrun.font.color.rgb = C_NAVY
        bdr = parse_xml(r'<w:pBdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:bottom w:val="single" w:sz="6" w:space="2" w:color="CBD5E1"/></w:pBdr>')
        h._p.get_or_add_pPr().append(bdr)

    # 4. Executive Summary
    add_section_heading('1. Executive Summary')
    p_exec = doc.add_paragraph()
    p_exec.paragraph_format.space_after = Pt(4)
    p_exec.paragraph_format.line_spacing = 1.15
    p_exec.add_run(
        'Lead optimization in medicinal chemistry is a difficult multi-objective problem: modifying an initial lead candidate '
        'to enhance drug-likeness (Quantitative Estimate of Drug-likeness, QED) while preserving its core structure (Tanimoto '
        'similarity >= 0.40), synthetic accessibility (Synthetic Accessibility Score, SAS <= 4.5), and pharmacological safety '
        '(Lipinski Rule of Five, PAINS/Brenk toxicophore filters). In this project, we formulate molecular lead optimization as '
        'a Markov Decision Process (MDP) over molecular graphs across 1,901 discrete chemical editing actions. We develop an '
        'Edge-Aware 3-Layer Graph Neural Network (GNN), a calibrated adaptive reward mechanism with an explicit STOP termination '
        'incentive, and a novel PPO-Guided Hybrid Top-5 local search optimizer that outperforms standard baselines.'
    )

    # 5. Core Architectural Contributions Table
    add_section_heading('2. Key Technical Contributions & Source Code Mapping')
    contrib_table = doc.add_table(rows=6, cols=3)
    contrib_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    contrib_headers = ['#', 'Contribution & Technical Mechanism', 'Primary Source File']
    header_row = contrib_table.rows[0]
    for j, h_text in enumerate(contrib_headers):
        c = header_row.cells[j]
        set_cell_background(c, '1B365D')
        p = c.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(255, 255, 255)

    contrib_data = [
        ('1', 'Dynamic Valence-Aware Action Masking: Pre-evaluates valence rules across 1,901 discrete chemical edits (Add, Remove, Substitute, Bond Change) to filter impossible chemical transitions before action selection.', 'src/environment/action_masking.py'),
        ('2', 'Edge-Aware 3-Layer GNN Encoder: 3-layer message-passing network with 4D chemical bond features (Single, Double, Triple, Aromatic) and dual-pooling (mean + max readout) combined with 6D scalar descriptors.', 'src/models/gnn_encoder_v2.py'),
        ('3', 'Calibrated Adaptive Reward Shaping: Dynamic headroom gain amplifying low-QED lead signals, smooth quadratic similarity barrier, SAS barrier, and a +2.0*Delta-QED STOP bonus for voluntary termination.', 'src/rewards/adaptive_reward.py'),
        ('4', 'PPO Actor-Critic Policy Optimization: Generalized Advantage Estimation (GAE), clipped surrogate objective, entropy regularization, and multi-tier curriculum sampling across starting-QED tiers.', 'src/agents/ppo_agent_v2.py'),
        ('5', 'PPO-Guided Hybrid Top-5 Search: Fast inductive policy ranking combined with lightweight 1-step lookahead across the top 5 actions, cutting property evaluations by 47x compared to exhaustive greedy search.', 'src/search/ppo_guided_search.py')
    ]

    widths_c = [Inches(0.35), Inches(4.35), Inches(2.0)]
    for i, (num, desc, src_file) in enumerate(contrib_data, start=1):
        row = contrib_table.rows[i]
        for j, (txt, w) in enumerate(zip([num, desc, src_file], widths_c)):
            cell = row.cells[j]
            cell.width = w
            set_cell_background(cell, 'F8FAFC' if i % 2 == 1 else 'FFFFFF')
            set_cell_margins(cell, 40, 40, 60, 60)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            r = p.add_run(txt)
            r.font.size = Pt(8.5)
            if j == 0:
                r.font.bold = True
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif j == 2:
                r.font.name = 'Consolas'
                r.font.size = Pt(8)
                r.font.color.rgb = C_BLUE

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 6. Benchmark Results Table
    add_section_heading('3. Empirical Benchmark Results (100 Scaffold-Separated Test Molecules)')
    p_bench = doc.add_paragraph()
    p_bench.paragraph_format.space_after = Pt(3)
    p_bench.add_run('Evaluation across 100 held-out test molecules stratified into 4 initial-QED tiers (25 molecules per tier):')
    p_bench.runs[0].font.size = Pt(9)

    res_table = doc.add_table(rows=7, cols=7)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_headers = ['Method', 'Mean Delta-QED', 'High-QED Success', 'Similarity', 'Violations', 'Evals/Mol', 'Latency']
    h_row = res_table.rows[0]
    for j, h_text in enumerate(res_headers):
        c = h_row.cells[j]
        set_cell_background(c, '1B365D')
        p = c.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(255, 255, 255)

    res_data = [
        ('Random Valid Edits', '-0.4654', '0.0%', '0.222', '95.0%', '9.51', '0.19s'),
        ('PPO Baseline (Unshaped)', '-0.3866', '2.0%', '0.264', '87.0%', '9.53', '0.21s'),
        ('PPO Proposed (Adaptive V1)', '-0.2054', '10.0%', '0.461', '46.0%', '9.56', '0.23s'),
        ('PPO Adaptive (V2 Edge-Aware)', '-0.1771', '9.0%', '0.715', '0.0%', '2.92', '0.10s'),
        ('PPO-Guided Search (Hybrid Top-5)', '+0.0183', '41.0%', '0.879', '0.0%', '5.68', '0.15s'),
        ('Greedy Hill Climbing (Oracle)', '+0.1076', '67.0%', '0.626', '0.0%', '268.22', '0.68s'),
    ]

    widths_r = [Inches(2.1), Inches(0.8), Inches(0.8), Inches(0.7), Inches(0.7), Inches(0.8), Inches(0.7)]
    for i, row_vals in enumerate(res_data, start=1):
        row = res_table.rows[i]
        is_highlight = 'Hybrid' in row_vals[0]
        for j, (val, w) in enumerate(zip(row_vals, widths_r)):
            cell = row.cells[j]
            cell.width = w
            if is_highlight:
                set_cell_background(cell, 'FEF3C7')
            else:
                set_cell_background(cell, 'F8FAFC' if i % 2 == 1 else 'FFFFFF')
            set_cell_margins(cell, 30, 30, 40, 40)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(8)
            if is_highlight:
                r.font.bold = True
                if j == 0:
                    r.font.color.rgb = RGBColor(180, 83, 9)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 7. Codebase Directory Guide
    add_section_heading('4. Repository Organization & File Directory')
    
    dir_table = doc.add_table(rows=10, cols=3)
    dir_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    dir_headers = ['Directory / Component', 'Key Files Included', 'Purpose & Deliverable']
    h_row = dir_table.rows[0]
    for j, h_text in enumerate(dir_headers):
        c = h_row.cells[j]
        set_cell_background(c, '1B365D')
        p = c.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    dir_data = [
        ('src/ (10 packages)', 'molecule_editor.py, gnn_encoder_v2.py, ppo_agent_v2.py, adaptive_reward.py, ppo_guided_search.py', 'Production Python library with zero legacy dependencies.'),
        ('notebooks/ (8 nbs)', '01_dataset_analysis.ipynb to 08_final_evaluation.ipynb', 'Complete end-to-end reproducible Jupyter workflow.'),
        ('code_output_figures/', '01_benchmark_500 to 10_reward_hacking_audit_chart.png', '10 publication-quality empirical figures matching the report.'),
        ('data/', 'data/raw/zinc_250k.csv, data/splits/train, val, test_benchmark', 'Raw ZINC-250K dataset and zero-leakage Bemis-Murcko splits.'),
        ('checkpoints/', 'baseline/, adaptive_v1/, adaptive_v2/, hybrid/ checkpoints', 'PyTorch serialized weights with SHA256 checksum manifest.'),
        ('results/', 'benchmark_500/, improvement_100/, ablation/, sensitivity/', 'Verified CSV outputs and metric logs for all experiments.'),
        ('reports/', 'ieee_access_report.tex, preliminary_project_report_v2.pdf', 'Complete academic IEEE Access LaTeX source paper and PDF.'),
        ('tests/', 'test_properties.py, test_gnn.py, test_ppo.py, test_reward.py (7 files)', '17 passing unit tests verifying full pipeline in <10 seconds.'),
        ('scripts/', 'reproduce_experiments.py, run_training.py, run_evaluation.py', 'Standalone CLI automation scripts for reproduction and training.')
    ]

    widths_d = [Inches(1.8), Inches(2.6), Inches(2.3)]
    for i, (d_name, d_files, d_desc) in enumerate(dir_data, start=1):
        row = dir_table.rows[i]
        for j, (txt, w) in enumerate(zip([d_name, d_files, d_desc], widths_d)):
            cell = row.cells[j]
            cell.width = w
            set_cell_background(cell, 'F8FAFC' if i % 2 == 1 else 'FFFFFF')
            set_cell_margins(cell, 30, 30, 50, 50)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            r = p.add_run(txt)
            r.font.size = Pt(8)
            if j == 0:
                r.font.bold = True
                r.font.color.rgb = C_NAVY
            elif j == 1:
                r.font.name = 'Consolas'
                r.font.size = Pt(7.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 8. Execution Commands
    add_section_heading('5. Quickstart & Verification Commands')
    cmd_box = doc.add_table(rows=1, cols=1)
    cmd_box.alignment = WD_TABLE_ALIGNMENT.CENTER
    cmd_cell = cmd_box.rows[0].cells[0]
    cmd_cell.width = Inches(6.7)
    set_cell_background(cmd_cell, '1E293B')
    set_cell_margins(cmd_cell, 60, 60, 100, 100)

    commands_text = (
        '# 1. Clone the repository\n'
        'git clone https://github.com/VuppalaShasank/Reinforcement-Learning-for-Multi-Objective-Drug-Likeness-Optimization-of-Small-Molecules.git\n'
        'cd Reinforcement-Learning-for-Multi-Objective-Drug-Likeness-Optimization-of-Small-Molecules\n\n'
        '# 2. Install dependencies\n'
        'pip install -r requirements.txt\n\n'
        '# 3. Run complete test suite (17/17 tests passing)\n'
        'python -m pytest\n\n'
        '# 4. Reproduce all benchmark results (Pilot 60, Benchmark 500, Improvement 100)\n'
        'python scripts/reproduce_experiments.py --suite all\n\n'
        '# 5. Regenerate all publication-quality empirical figures\n'
        'python scripts/generate_academic_figures.py\n\n'
        '# 6. Train the Edge-Aware PPO agent with dynamic curriculum\n'
        'python scripts/run_training.py --episodes 100 --curriculum'
    )

    cp_cmd = cmd_cell.paragraphs[0]
    cp_cmd.paragraph_format.space_before = Pt(0)
    cp_cmd.paragraph_format.space_after = Pt(0)
    cp_cmd.paragraph_format.line_spacing = 1.15
    rcmd = cp_cmd.add_run(commands_text)
    rcmd.font.name = 'Consolas'
    rcmd.font.size = Pt(8)
    rcmd.font.color.rgb = RGBColor(241, 245, 249)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # 9. Academic Citation
    add_section_heading('6. Academic Citation')
    p_cite = doc.add_paragraph()
    p_cite.paragraph_format.line_spacing = 1.1
    p_cite.paragraph_format.space_after = Pt(0)
    rcite = p_cite.add_run(
        'Vuppala, Shasank. "Reinforcement Learning for Multi-Objective Drug-Likeness Optimization of Small Molecules: '
        'From Adaptive Policy Gradients to Policy-Guided Local Search." Department of Computer Science and Engineering (Artificial Intelligence), '
        'Amrita Vishwa Vidyapeetham, Amritapuri, Kerala, India (2026).'
    )
    rcite.font.size = Pt(9)
    rcite.font.italic = True

    doc.save(output_path)
    print(f'Document successfully created: {output_path}')

if __name__ == '__main__':
    create_document('Project_Source_Code_and_Details.docx')
    create_document('reports/Project_Source_Code_and_Details.docx')
