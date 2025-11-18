# CLAUDE.md - AI Assistant Guide for Hugging Face Daily Papers Bot

> **Last Updated**: 2025-11-18
> **Purpose**: Comprehensive guide for AI assistants working with this repository

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Key Workflows](#key-workflows)
4. [Development Conventions](#development-conventions)
5. [Custom Commands](#custom-commands)
6. [Configuration](#configuration)
7. [Output Formats](#output-formats)
8. [Common Tasks](#common-tasks)
9. [Git Workflow](#git-workflow)
10. [Best Practices](#best-practices)

---

## 📚 Project Overview

### Purpose
Automated bot that fetches, analyzes, and summarizes top AI/ML papers from Hugging Face Daily Papers in Korean. The bot provides in-depth analysis suitable for researchers, ML engineers, and AI enthusiasts.

### Key Features
- **Daily Analysis**: Automatically fetch and analyze top 3 papers daily
- **Monthly Collections**: Aggregate and analyze top 10 papers per month
- **Deep Analysis**: 11-point comprehensive analysis framework per paper
- **Korean Summaries**: Professional, accessible Korean translations
- **Automated Workflow**: GitHub Actions for daily execution
- **Custom Slash Command**: `/해` for Claude Code integration

### Technology Stack
- **Python 3.11+**: Core language
- **Hugging Face API**: Paper data source (`https://huggingface.co/api/daily_papers`)
- **Anthropic Claude API**: AI-powered analysis (claude-3-5-sonnet-20241022)
- **GitHub Actions**: Automation and scheduling
- **Claude Code**: Custom slash commands and workflows

---

## 📁 Repository Structure

```
for_paper/
├── .claude/
│   └── commands/
│       └── 해.md              # Custom slash command for daily analysis
├── .github/
│   └── workflows/
│       └── daily-papers.yml   # Daily automation workflow
├── src/
│   ├── main.py                # Main entry point for daily reports
│   ├── fetch_papers.py        # Hugging Face API client
│   ├── summarize_papers.py    # Claude API integration
│   └── collect_monthly_papers.py  # Monthly collection script
├── outputs/
│   ├── daily_papers_YYYY-MM-DD.md   # Daily reports
│   └── monthly_papers_YYYY_MM.md    # Monthly reports
├── data/
│   └── papers_YYYY_MM.json    # Monthly paper data (JSON)
├── .env.example               # Environment variable template
├── .gitignore                 # Git ignore patterns
├── requirements.txt           # Python dependencies
├── README.md                  # User-facing documentation (Korean)
└── CLAUDE.md                  # This file (AI assistant guide)
```

### Key Files Explained

#### **src/fetch_papers.py**
- `PaperFetcher` class for API interactions
- Fetches papers from Hugging Face Daily Papers API
- Methods:
  - `fetch_daily_papers(date, limit)`: Get papers for specific date
  - `get_top_papers(count)`: Get top N papers by upvotes
  - `format_paper_info(paper)`: Standardize paper data structure

#### **src/summarize_papers.py**
- `PaperSummarizer` class for Claude API integration
- Generates Korean summaries using Claude 3.5 Sonnet
- Methods:
  - `create_summary_prompt(paper)`: Build analysis prompt
  - `summarize_paper(paper)`: Single paper analysis
  - `summarize_papers(papers)`: Batch processing

#### **src/main.py**
- `DailyPaperReport` class orchestrating the pipeline
- Methods:
  - `run(top_n)`: Complete pipeline execution
  - `generate_markdown(papers, date)`: Format as markdown
  - `save_report(markdown, date)`: Write to file
  - `update_readme(latest_date)`: Update README with latest link

#### **src/collect_monthly_papers.py**
- `MonthlyPaperCollector` class for monthly aggregation
- Collects all papers for a month and identifies top 10
- Deduplicates by arxiv_id, keeps highest upvotes
- Saves to JSON in `data/` directory

---

## 🔄 Key Workflows

### 1. Daily Paper Analysis (via `/해` command)

**Trigger**: User types `/해` or `해` in Claude Code

**Process**:
1. Fetch today's top 3 papers from Hugging Face API
2. For each paper, extract:
   - Title, authors, ArXiv ID, abstract, upvotes
   - URLs (ArXiv and Hugging Face)
3. Perform deep analysis using 11-point framework (see Output Formats)
4. Generate markdown report in Korean
5. Save to `outputs/daily_papers_YYYY-MM-DD.md`
6. Commit and push with message: `📚 Daily papers update: YYYY-MM-DD`

**Expected Duration**: 2-5 minutes depending on paper length

### 2. Monthly Collection (Python script)

**Trigger**: Manual execution of `src/collect_monthly_papers.py`

**Process**:
1. Iterate through each day of the target month
2. Fetch papers for each day (with 0.5s delay between requests)
3. Deduplicate by arxiv_id (keep paper with most upvotes)
4. Sort by upvotes and select top 10
5. Save to `data/papers_YYYY_MM.json`
6. Can be used as input for monthly summary generation

### 3. GitHub Actions Automation

**Schedule**: Daily at 09:00 UTC (18:00 KST)

**Steps**:
1. Checkout repository
2. Setup Python 3.11
3. Install dependencies from requirements.txt
4. Run `src/main.py` with `ANTHROPIC_API_KEY` from secrets
5. Commit and push outputs/ and README.md if changed
6. Uses git user: `github-actions[bot]`

---

## 💻 Development Conventions

### Code Style
- **Python**: Follow PEP 8 conventions
- **Docstrings**: Use Google-style docstrings for all classes and functions
- **Type Hints**: Include type hints for function parameters and returns
- **Error Handling**: Use try-except blocks with informative error messages
- **Logging**: Print progress messages with emoji prefixes for clarity

### Naming Conventions
- **Classes**: PascalCase (e.g., `PaperFetcher`, `DailyPaperReport`)
- **Functions/Methods**: snake_case (e.g., `get_top_papers`, `format_paper_info`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `API_URL`)
- **Files**: snake_case (e.g., `fetch_papers.py`)

### File Naming Patterns
- Daily reports: `daily_papers_YYYY-MM-DD.md`
- Monthly reports: `monthly_papers_YYYY_MM.md`
- Monthly data: `papers_YYYY_MM.json`

### Paper Data Structure
Standard format across all modules:
```python
{
    "title": str,
    "paper_id": str,          # Hugging Face paper ID
    "arxiv_id": str,          # ArXiv ID (e.g., "2001.08361")
    "url": str,               # ArXiv URL
    "huggingface_url": str,   # Hugging Face URL
    "authors": list[str],
    "abstract": str,
    "upvotes": int,
    "published_at": str,
    "thumbnail": str,
    "summary": str            # Added after Claude analysis
}
```

---

## 🎯 Custom Commands

### `/해` - Daily Paper Analysis

**Location**: `.claude/commands/해.md`

**Description**: Hugging Face Daily Papers Top 3를 심층 분석하고 상세하게 정리

**Usage**:
```
/해
```
or simply:
```
해
```

**What It Does**:
1. Fetches top 3 papers from Hugging Face Daily Papers
2. Performs 11-point deep analysis per paper
3. Generates comprehensive Korean markdown report
4. Commits and pushes to repository

**Analysis Framework** (11 points per paper):
1. **한 줄 요약** (TL;DR) - One-sentence essence
2. **연구 배경과 동기** - Problem definition and motivation
3. **핵심 아이디어** - Core innovation and differentiation
4. **기술적 접근** - Technical methods and architecture
5. **주요 기여점** - Academic contributions (3-4 bullets)
6. **실험 및 결과** - Datasets and performance metrics
7. **강점과 영향력** - Strengths and potential impact
8. **한계점 및 고려사항** - Limitations and caveats
9. **응용 가능성** - Real-world applications
10. **관련 연구 맥락** - Research context and trends
11. **핵심 키워드** - 5-7 technical keywords

**Important Notes**:
- Analysis must be **깊이 있는** (in-depth), not superficial summaries
- Korean should be professional yet accessible
- Use emojis to enhance readability
- Identify overall daily trends across all papers
- Suggest target audience for each paper

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file based on `.env.example`:

```bash
# Anthropic API Key for Claude
ANTHROPIC_API_KEY=your_api_key_here
```

**Required**: ANTHROPIC_API_KEY is essential for paper summarization

**Note**: GitHub Actions uses `ANTHROPIC_API_KEY` from repository secrets

### Dependencies

From `requirements.txt`:
```
requests==2.31.0        # HTTP library for API calls
anthropic==0.18.1       # Anthropic Claude API client
python-dotenv==1.0.1    # Environment variable management
```

Install with:
```bash
pip install -r requirements.txt
```

### API Configuration

**Hugging Face API**:
- Endpoint: `https://huggingface.co/api/daily_papers`
- No authentication required
- Rate limit: Be respectful, use 0.5s delay for bulk requests
- Parameters:
  - `date`: YYYY-MM-DD format (optional, defaults to today)
  - `limit`: Max papers to return (default: 100)

**Claude API**:
- Model: `claude-3-5-sonnet-20241022`
- Max tokens: 2000 (for summaries)
- Temperature: 0.7 (balanced creativity and consistency)
- Current version: anthropic==0.18.1

---

## 📄 Output Formats

### Daily Report Structure

File: `outputs/daily_papers_YYYY-MM-DD.md`

```markdown
# 📚 Daily Papers - YYYY년 MM월 DD일

> Hugging Face Daily Papers의 Top 3 논문을 심층 분석한 리포트입니다.
>
> 💡 **오늘의 하이라이트**: [Overall trend 1-2 sentences]

---

## 🏆 1위: [Paper Title]

> **TL;DR**: [One-sentence summary]

### 📊 기본 정보

- **저자**: [Author list]
- **소속**: [Institution if identifiable]
- **ArXiv**: [ArXiv URL]
- **Hugging Face**: [HF URL]
- **Upvotes**: XX 👍

---

### 🎯 연구 배경과 동기
[Content]

### 💡 핵심 아이디어
[Content]

### 🔧 기술적 접근
[Content]

### 🌟 주요 기여점
- [Contribution 1]
- [Contribution 2]
- [Contribution 3]

### 📈 실험 및 결과
[Content]

### 💪 강점과 영향력
[Content]

### ⚠️ 한계점 및 고려사항
[Content]

### 🚀 응용 가능성
[Content]

### 🔗 관련 연구 맥락
[Content]

### 🏷️ 핵심 키워드
`keyword1` `keyword2` `keyword3` `keyword4` `keyword5`

---

[Papers 2 and 3 follow same format]

---

## 📌 전체 요약

### 오늘의 주요 트렌드
- [Trend 1]
- [Trend 2]

### 주목해야 할 기술
- [Technology 1]
- [Technology 2]

### 추천 독자
- **[Paper 1]**: [Who should read this]
- **[Paper 2]**: [Who should read this]
- **[Paper 3]**: [Who should read this]

---

*Generated with 🤖 AI Analysis on YYYY-MM-DD HH:MM:SS*
*Powered by Claude & Hugging Face*
```

### Monthly Report Structure

File: `outputs/monthly_papers_YYYY_MM.md`

Similar structure but for top 10 papers of the month, with additional monthly trend analysis.

### Monthly Data JSON

File: `data/papers_YYYY_MM.json`

```json
[
  {
    "title": "Paper Title",
    "paper_id": "12345",
    "arxiv_id": "2001.08361",
    "url": "https://arxiv.org/abs/2001.08361",
    "huggingface_url": "https://huggingface.co/papers/12345",
    "authors": ["Author 1", "Author 2"],
    "abstract": "Full abstract text...",
    "upvotes": 342,
    "published_at": "2025-01-15T10:00:00Z",
    "thumbnail": "https://..."
  }
]
```

---

## 🛠️ Common Tasks

### Task 1: Run Daily Analysis Manually

**Using Claude Code**:
```
/해
```

**Using Python directly**:
```bash
cd src
python main.py
```

**Requirements**:
- ANTHROPIC_API_KEY must be set in `.env`
- Internet connection for API access

### Task 2: Collect Monthly Papers

```bash
cd src
python collect_monthly_papers.py
```

**Note**: Modify the script's `main()` function to target specific months:
```python
for month in range(1, 11):  # Jan to Oct
```

### Task 3: Test Paper Fetching Only

```bash
cd src
python fetch_papers.py
```

Shows top 3 papers without generating summaries.

### Task 4: Test Summarization

```bash
cd src
python summarize_papers.py
```

Tests Claude API connection with a sample paper.

### Task 5: Generate Custom Analysis

To analyze a different number of papers, modify in Claude Code prompt or Python:

**Python**:
```python
report_generator = DailyPaperReport()
report_generator.run(top_n=5)  # Analyze top 5 instead
```

**Claude Code**: Modify the `/해` command's instructions

### Task 6: Update Dependencies

```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

**Important**: Test thoroughly after updates, especially `anthropic` library.

---

## 🔀 Git Workflow

### Branch Strategy

**Main Branch**: All production-ready code
**Feature Branches**: Follow pattern `claude/claude-md-{session-id}`

### Commit Message Conventions

**Daily Updates**:
```
📚 Daily papers update: YYYY-MM-DD
```

**Monthly Collections**:
```
📚 Add YYYY-MM monthly paper collection
```

**Monthly Reports**:
```
📚 Add Month YYYY Top 10 papers analysis (complete)
```
or for partial:
```
📚 Add Month YYYY Top 10 papers analysis (N/10 completed)
```

**Infrastructure Changes**:
```
🔧 Add monthly paper collection infrastructure
```

**Bug Fixes**:
```
🐛 Fix issue with paper deduplication
```

**Documentation**:
```
📝 Update README with new features
```

### Commit Workflow

1. **Stage changes**:
   ```bash
   git add outputs/
   git add README.md  # if updated
   ```

2. **Commit with appropriate message**:
   ```bash
   git commit -m "📚 Daily papers update: 2025-11-18"
   ```

3. **Push to remote**:
   ```bash
   git push -u origin <branch-name>
   ```

**Important**: Branch name must start with `claude/` and end with session ID for proper permissions.

### Files to Commit

**Always commit**:
- `outputs/*.md` - Generated reports
- `data/*.json` - Collected paper data
- `README.md` - If updated with latest report link

**Never commit**:
- `.env` - Contains API keys (in .gitignore)
- `__pycache__/` - Python cache files
- `.venv/`, `venv/` - Virtual environments

---

## ✅ Best Practices

### For AI Assistants Working on This Repo

1. **Always Read Before Writing**
   - Use Read tool before Edit/Write on existing files
   - Check current state of outputs/ directory before generating new reports
   - Verify existing reports to maintain consistency

2. **Follow the Analysis Framework**
   - All 11 points must be covered for each paper
   - Maintain depth and quality - avoid superficial summaries
   - Use professional Korean with appropriate technical terms
   - Include emojis as per the template

3. **Error Handling**
   - Check API responses before processing
   - Handle missing abstracts gracefully
   - Provide informative error messages if summarization fails
   - Never create empty or incomplete reports

4. **File Operations**
   - Use absolute paths for all file operations
   - Ensure `outputs/` and `data/` directories exist before writing
   - Follow exact naming conventions for output files
   - Preserve file encoding as UTF-8

5. **API Usage**
   - Verify ANTHROPIC_API_KEY is set before starting
   - Use appropriate model (claude-3-5-sonnet-20241022)
   - Handle rate limits and API errors gracefully
   - Add delays (0.5s) when making bulk API requests

6. **Git Operations**
   - Always check git status before committing
   - Use exact commit message formats
   - Push to correct branch with proper naming
   - Handle network errors with retry logic (up to 4 times with exponential backoff)

7. **Output Quality**
   - Ensure markdown is properly formatted
   - Verify all links are valid
   - Check Korean text for clarity and professionalism
   - Include all required sections (highlights, summary, etc.)

8. **Date Handling**
   - Use YYYY-MM-DD format consistently
   - Convert to Korean format (YYYY년 MM월 DD일) for display
   - Use current date unless specified otherwise
   - Handle timezones appropriately (KST for reports)

### Code Modification Guidelines

**When adding features**:
- Follow existing code patterns and conventions
- Add type hints and docstrings
- Update this CLAUDE.md file
- Test with sample data before production use

**When fixing bugs**:
- Identify root cause before patching
- Add error handling to prevent recurrence
- Update tests if applicable
- Document the fix in commit message

**When updating dependencies**:
- Check for breaking changes in changelogs
- Test all workflows after update
- Update version numbers in requirements.txt
- Note any API changes that affect the code

### Testing Checklist

Before pushing changes:
- [ ] Code runs without errors
- [ ] API calls succeed
- [ ] Output files are generated correctly
- [ ] Markdown formatting is valid
- [ ] Korean text is clear and professional
- [ ] All required sections are present
- [ ] Links are functional
- [ ] Commit message follows conventions
- [ ] .env file is not committed

---

## 🔍 Troubleshooting

### Common Issues

**Issue**: "ANTHROPIC_API_KEY not found"
- **Solution**: Create `.env` file with valid API key
- **Check**: File is in repository root, not in src/

**Issue**: "No papers found"
- **Solution**: Check Hugging Face API status
- **Alternative**: Try different date or check internet connection

**Issue**: API rate limit errors
- **Solution**: Add delay between requests (already implemented: 0.5s)
- **For bulk operations**: Consider running during off-peak hours

**Issue**: Git push fails with 403
- **Solution**: Ensure branch name starts with `claude/` and ends with valid session ID
- **Check**: Branch naming convention is correct

**Issue**: Korean text encoding errors
- **Solution**: Always use `encoding='utf-8'` for file operations
- **Verify**: Files are saved with UTF-8 encoding

**Issue**: Markdown formatting broken
- **Solution**: Follow exact template structure
- **Validate**: Use markdown linter or preview

---

## 📚 Additional Resources

### API Documentation
- [Hugging Face Daily Papers API](https://huggingface.co/api/daily_papers)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Claude Code Documentation](https://docs.claude.com/en/docs/claude-code)

### Project Links
- **Repository**: [thisiskorea/for_paper](https://github.com/thisiskorea/for_paper)
- **Issues**: [GitHub Issues](https://github.com/thisiskorea/for_paper/issues)
- **Hugging Face Daily Papers**: [https://huggingface.co/papers](https://huggingface.co/papers)

### Related Documentation
- README.md - User-facing documentation (Korean)
- .claude/commands/해.md - Slash command specification
- .github/workflows/daily-papers.yml - Automation workflow

---

## 🤖 AI Assistant Quick Reference

### When User Says "/해" or "해"

1. Fetch today's top 3 papers
2. Analyze each with 11-point framework
3. Generate markdown report (Korean)
4. Save to `outputs/daily_papers_YYYY-MM-DD.md`
5. Commit: `📚 Daily papers update: YYYY-MM-DD`
6. Push to remote

### Key Files to Remember

- Analysis logic: `.claude/commands/해.md`
- Paper fetching: `src/fetch_papers.py`
- Summarization: `src/summarize_papers.py`
- Pipeline: `src/main.py`
- Output location: `outputs/`

### Critical Conventions

- Date format: YYYY-MM-DD
- Korean format: YYYY년 MM월 DD일
- Commit emoji: 📚 for papers, 🔧 for infrastructure
- Always UTF-8 encoding
- Always include all 11 analysis points

---

**Remember**: This repository is about providing high-quality, in-depth analysis of AI research papers in accessible Korean. Quality over quantity, depth over breadth.

---

*Last updated: 2025-11-18 by Claude Code*
