"""
Collect monthly top papers from Hugging Face Daily Papers
"""
import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict
import time


class MonthlyPaperCollector:
    """Collects top papers for each month"""

    API_URL = "https://huggingface.co/api/daily_papers"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def get_papers_for_date(self, date_str: str) -> List[Dict]:
        """
        Get papers for a specific date

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            List of papers
        """
        try:
            response = self.session.get(
                self.API_URL,
                params={'date': date_str, 'limit': 100},
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get papers for {date_str}: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error fetching papers for {date_str}: {e}")
            return []

    def get_month_papers(self, year: int, month: int) -> List[Dict]:
        """
        Get all papers for a specific month

        Args:
            year: Year (e.g., 2025)
            month: Month (1-12)

        Returns:
            List of papers with deduplicated by arxiv_id
        """
        print(f"\n📅 Collecting papers for {year}-{month:02d}...")

        # Get first and last day of month
        first_day = datetime(year, month, 1)
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)

        all_papers = {}  # Use dict to deduplicate by arxiv_id

        current_day = first_day
        while current_day <= last_day:
            date_str = current_day.strftime('%Y-%m-%d')
            print(f"  Fetching {date_str}...", end=" ")

            papers = self.get_papers_for_date(date_str)
            print(f"Found {len(papers)} papers")

            for paper in papers:
                arxiv_id = paper.get('arxivId', '')
                if arxiv_id and arxiv_id not in all_papers:
                    all_papers[arxiv_id] = paper
                elif arxiv_id and arxiv_id in all_papers:
                    # Keep the one with more upvotes
                    if paper.get('upvotes', 0) > all_papers[arxiv_id].get('upvotes', 0):
                        all_papers[arxiv_id] = paper

            current_day += timedelta(days=1)
            time.sleep(0.5)  # Be nice to the API

        paper_list = list(all_papers.values())
        print(f"\n✅ Total unique papers for {year}-{month:02d}: {len(paper_list)}")
        return paper_list

    def get_top_papers(self, papers: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        Get top N papers sorted by upvotes

        Args:
            papers: List of papers
            top_n: Number of top papers to return

        Returns:
            List of top N papers
        """
        sorted_papers = sorted(papers, key=lambda x: x.get('upvotes', 0), reverse=True)
        return sorted_papers[:top_n]

    def format_paper_info(self, paper: Dict) -> Dict:
        """Format paper information"""
        return {
            "title": paper.get("title", "No title"),
            "paper_id": paper.get("paper", {}).get("id", ""),
            "arxiv_id": paper.get("arxivId", ""),
            "url": f"https://arxiv.org/abs/{paper.get('arxivId', '')}" if paper.get('arxivId') else "",
            "huggingface_url": f"https://huggingface.co/papers/{paper.get('paper', {}).get('id', '')}" if paper.get('paper', {}).get('id') else "",
            "authors": paper.get("authors", []),
            "abstract": paper.get("abstract", "No abstract available"),
            "upvotes": paper.get("upvotes", 0),
            "published_at": paper.get("publishedAt", ""),
            "thumbnail": paper.get("thumbnail", ""),
        }

    def save_monthly_data(self, year: int, month: int, papers: List[Dict], output_dir: str = "data"):
        """Save monthly paper data to JSON"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{output_dir}/papers_{year}_{month:02d}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved to {filename}")


def main():
    """Main function to collect papers for 2025 Jan-Oct"""
    collector = MonthlyPaperCollector()

    for month in range(1, 11):  # Jan to Oct
        print(f"\n{'='*60}")
        print(f"📚 Processing 2025-{month:02d}")
        print(f"{'='*60}")

        try:
            # Get all papers for the month
            all_papers = collector.get_month_papers(2025, month)

            # Get top 10
            top_10 = collector.get_top_papers(all_papers, top_n=10)

            # Format papers
            formatted_papers = [collector.format_paper_info(p) for p in top_10]

            # Save to JSON
            collector.save_monthly_data(2025, month, formatted_papers)

            # Display summary
            print(f"\n🏆 Top 10 papers for 2025-{month:02d}:")
            for i, paper in enumerate(formatted_papers, 1):
                print(f"  {i}. {paper['title'][:60]}... ({paper['upvotes']} upvotes)")

        except Exception as e:
            print(f"❌ Error processing 2025-{month:02d}: {e}")
            continue

    print(f"\n{'='*60}")
    print("✨ Data collection complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
