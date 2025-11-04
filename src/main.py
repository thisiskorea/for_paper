"""
Main script to fetch, summarize, and generate daily paper reports
"""
import os
import sys
from datetime import datetime
from pathlib import Path

from fetch_papers import PaperFetcher
from summarize_papers import PaperSummarizer


class DailyPaperReport:
    """Generates daily paper reports"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fetcher = PaperFetcher()
        self.summarizer = None

    def initialize_summarizer(self):
        """Initialize the summarizer (only when needed)"""
        if self.summarizer is None:
            try:
                self.summarizer = PaperSummarizer()
            except ValueError as e:
                print(f"Warning: {e}")
                print("Summaries will not be generated.")
                return False
        return True

    def generate_markdown(self, papers: list, date: str) -> str:
        """
        Generate markdown report from papers

        Args:
            papers: List of papers with summaries
            date: Date string in YYYY-MM-DD format

        Returns:
            Markdown formatted report
        """
        # Convert date format for display
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%Y년 %m월 %d일")

        markdown = f"""# Daily Papers - {formatted_date}

> Hugging Face Daily Papers의 Top 3 논문을 분석한 리포트입니다.

---

"""

        for i, paper in enumerate(papers, 1):
            markdown += f"""
## {i}. {paper['title']}

**저자**: {', '.join(paper['authors'][:5])}{' 외' if len(paper['authors']) > 5 else ''}

**링크**:
- [ArXiv]({paper['url']})
- [Hugging Face]({paper['huggingface_url']})

**Upvotes**: {paper['upvotes']} 👍

### 요약

{paper.get('summary', '요약을 생성할 수 없습니다.')}

---

"""

        # Add footer
        markdown += f"""
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by [HuggingFace Daily Papers Bot](https://github.com/thisiskorea/for_paper)*
"""

        return markdown

    def save_report(self, markdown: str, date: str):
        """
        Save markdown report to file

        Args:
            markdown: Markdown content
            date: Date string in YYYY-MM-DD format
        """
        filename = f"daily_papers_{date}.md"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown)

        print(f"\n✅ Report saved to: {filepath}")
        return filepath

    def update_readme(self, latest_date: str):
        """
        Update README.md with link to latest report

        Args:
            latest_date: Date of the latest report
        """
        readme_path = Path("README.md")

        # If README doesn't exist, it will be created separately
        if not readme_path.exists():
            return

        latest_report_link = f"outputs/daily_papers_{latest_date}.md"

        # Read existing README
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Update latest report link
        if "## Latest Report" in content:
            # Update existing link
            import re
            pattern = r"(## Latest Report\n\n).*?(\n\n##|\Z)"
            replacement = f"\\1[📄 {latest_date} Daily Papers]({latest_report_link})\\2"
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        else:
            # Add new section
            latest_section = f"\n\n## Latest Report\n\n[📄 {latest_date} Daily Papers]({latest_report_link})\n"
            content += latest_section

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ README.md updated with latest report")

    def run(self, top_n: int = 3):
        """
        Run the complete pipeline

        Args:
            top_n: Number of top papers to process (default: 3)
        """
        print("=" * 60)
        print("🤖 Hugging Face Daily Papers Bot")
        print("=" * 60)

        # Get current date
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"\n📅 Date: {today}")

        # Fetch papers
        print(f"\n📚 Fetching top {top_n} papers from Hugging Face...")
        papers = self.fetcher.get_top_papers(count=top_n)

        if not papers:
            print("❌ No papers found!")
            sys.exit(1)

        print(f"✅ Found {len(papers)} papers")

        # Format papers
        formatted_papers = [self.fetcher.format_paper_info(paper) for paper in papers]

        # Display papers
        print("\n📋 Papers to summarize:")
        for i, paper in enumerate(formatted_papers, 1):
            print(f"  {i}. {paper['title'][:70]}...")

        # Summarize papers
        if self.initialize_summarizer():
            print(f"\n🤖 Generating summaries using Claude...")
            summarized_papers = self.summarizer.summarize_papers(formatted_papers)
        else:
            print("\n⚠️  Skipping summaries (API key not configured)")
            summarized_papers = formatted_papers

        # Generate markdown
        print(f"\n📝 Generating markdown report...")
        markdown = self.generate_markdown(summarized_papers, today)

        # Save report
        filepath = self.save_report(markdown, today)

        # Update README
        self.update_readme(today)

        print("\n" + "=" * 60)
        print("✨ Done!")
        print("=" * 60)

        return filepath


def main():
    """Main entry point"""
    report_generator = DailyPaperReport()
    report_generator.run(top_n=3)


if __name__ == "__main__":
    main()
