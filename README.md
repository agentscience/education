
# Urban-Rural Disparities in Returns to Education: Evidence from CHIP 2018

## Project Overview

This repository contains the implementation and analysis for our research on heterogeneous returns to education in China, utilizing the China Household Income Project (CHIP) 2018 dataset. The study employs Mincer equation methodology and advanced econometric techniques to examine education-income relationships across different demographic groups.

**Submitted to 1st Open Conference on AI Agents for Science (agents4science 2025)**

## Research Objectives

This study addresses three central research questions:

1. **Current Returns to Education**: What are the contemporary returns to education in China using the latest CHIP 2018 data? We provide updated estimates employing standard Mincer equation methodology.

2. **Heterogeneity Analysis**: How do returns vary across different demographic groups? We examine heterogeneity along multiple dimensions including:
   - Urban-rural residence
   - Gender disparities
   - Age cohorts
   - Income quantiles

3. **Education-Inequality Nexus**: What is the relationship between education and income inequality? Through quantile regression analysis, we investigate whether education serves as an equalizing force or exacerbates existing disparities.

## Methodology

### Two-Stage Analysis Pipeline

#### Stage 1: Data Processing and Preparation
- Clean and merge CHIP 2018 household and individual datasets
- Handle missing data and outliers systematically
- Generate key variables (education years, work experience, regional indicators)
- Create stratified samples by urban-rural status

#### Stage 2: Econometric Analysis
- **Basic Mincer Equation**: Estimate baseline returns to education
- **Extended Models**: Include controls for gender, region, industry
- **Heterogeneity Analysis**: Interaction effects and subgroup analysis
- **Robustness Checks**: Alternative specifications and sensitivity tests

## Repository Structure

### Core Analysis Scripts
- `analyze_real_chip_data.py` - Main CHIP 2018 data analysis pipeline
- `chip2018_comprehensive_analysis.py` - Comprehensive econometric analysis
- `chip_data_analysis_corrected.py` - Corrected data processing module
- `verify_real_chip_data.py` - Data validation and quality checks

### Visualization Scripts
- `generate_academic_figures.py` - Academic publication-ready figures
- `generate_all_six_charts.py` - Generate six core analysis charts
- `generate_dissertation_charts.py` - Dissertation-specific visualizations
- `generate_real_data_figures.py` - Real data visualization suite

### LaTeX Documentation
- `chip2018_agents4science.tex` - Main LaTeX manuscript source
- `chip2018_agents4science.pdf` - Compiled PDF manuscript
- `agents4science_2025.sty` - Conference style file
- `references.bib` - Bibliography database

### Figures and Charts

#### `chart/` - Primary Analysis Figures
- `figure1_sample_distribution.png` - Sample characteristics distribution
- `figure2_returns_comparison.png` - Returns to education comparison
- `figure3_income_education.png` - Income-education relationship
- `figure4_heterogeneity.png` - Heterogeneity analysis results
- `figure5_income_distribution.png` - Income distribution patterns
- `figure6_regression_diagnostics.png` - Regression diagnostic plots

#### `figures/` - Supplementary Materials
- Comparative analyses
- Robustness checks
- Additional diagnostic plots

### Final Outputs
- `final/` - Conference submissions and final papers
  - Published papers with appendices
  - Replication packages

### Documentation
- `CHIP2018_variable_verification.md` - Variable definitions and verification

## Key Findings

### 1. Significant Urban-Rural Disparities
- **Rural Returns**: 8.3% per year of education
- **Urban Returns**: 5.5% per year of education
- **Gap**: 2.8 percentage points (statistically significant at 1% level)

### 2. Heterogeneous Effects Across Groups
- **Gender**: Male returns exceed female by 1.2 percentage points
- **Age Cohorts**: Younger cohorts show higher returns (9.1% vs 6.2% for older)
- **Income Quantiles**: Returns decrease with income level (Q1: 10.2%, Q5: 4.8%)

### 3. Policy Implications
- Education investment shows highest returns for:
  - Rural residents
  - Lower-income households
  - Younger populations
- Targeted education policies could reduce income inequality

## Technical Requirements

### Python Dependencies
```
python>=3.8
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.12.0
scikit-learn>=0.24.0
```

### LaTeX Compilation
- Full LaTeX distribution (TeX Live or MiKTeX)
- pdflatex command-line tools
- BibTeX support

## Usage Instructions

### 1. Data Analysis
```bash
# Run main analysis pipeline
python analyze_real_chip_data.py

# Comprehensive econometric analysis
python chip2018_comprehensive_analysis.py
```

### 2. Generate Figures
```bash
# Generate all six core charts
python generate_all_six_charts.py

# Create academic figures
python generate_academic_figures.py
```

### 3. Compile LaTeX Document
```bash
# Full compilation with bibliography
pdflatex chip2018_agents4science.tex
bibtex chip2018_agents4science
pdflatex chip2018_agents4science.tex
pdflatex chip2018_agents4science.tex
```

## Data Source

**China Household Income Project (CHIP) 2018**
- Survey Year: 2018
- Sample Size: ~15,000 individuals
- Coverage: Urban and rural areas across China
- Key Variables: Income, education, demographics, employment

## Reproducibility

All analyses are fully reproducible with:
- Raw data access (requires CHIP 2018 license)
- Python scripts for data processing
- Complete econometric specifications
- Random seeds set for consistency


## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- China Household Income Project team for data access
- Anonymous reviewers for valuable feedback


---

*Last Updated: Sep 2025*  
*Version: 1.0.0*  