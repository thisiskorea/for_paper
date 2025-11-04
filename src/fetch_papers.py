"""
Fetch daily papers from Hugging Face API
"""
import requests
import json
from datetime import datetime
from typing import List, Dict, Optional


class PaperFetcher:
    """Fetches papers from Hugging Face Daily Papers API"""

    API_URL = "https://huggingface.co/api/daily_papers"

    def __init__(self):
        self.session = requests.Session()

    def fetch_daily_papers(self, date: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Fetch daily papers from Hugging Face

        Args:
            date: Date in YYYY-MM-DD format. If None, gets today's papers
            limit: Maximum number of papers to fetch

        Returns:
            List of paper dictionaries
        """
        params = {"limit": limit}
        if date:
            params["date"] = date

        try:
            response = self.session.get(self.API_URL, params=params)
            response.raise_for_status()
            papers = response.json()
            return papers
        except requests.exceptions.RequestException as e:
            print(f"Error fetching papers: {e}")
            return []

    def get_top_papers(self, count: int = 3) -> List[Dict]:
        """
        Get top N papers from today's daily papers

        Args:
            count: Number of top papers to return (default: 3)

        Returns:
            List of top paper dictionaries
        """
        papers = self.fetch_daily_papers()

        if not papers:
            return []

        # Papers are already sorted by popularity/upvotes from the API
        # We just need to take the first N papers
        top_papers = papers[:count]

        return top_papers

    def format_paper_info(self, paper: Dict) -> Dict:
        """
        Format paper information into a clean structure

        Args:
            paper: Raw paper data from API

        Returns:
            Formatted paper information
        """
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


def main():
    """Test the paper fetcher"""
    fetcher = PaperFetcher()

    print("Fetching top 3 papers from Hugging Face Daily Papers...")
    top_papers = fetcher.get_top_papers(count=3)

    if top_papers:
        print(f"\nFound {len(top_papers)} papers:")
        for i, paper in enumerate(top_papers, 1):
            formatted = fetcher.format_paper_info(paper)
            print(f"\n{i}. {formatted['title']}")
            print(f"   Upvotes: {formatted['upvotes']}")
            print(f"   ArXiv: {formatted['url']}")
            print(f"   Authors: {', '.join(formatted['authors'][:3])}{'...' if len(formatted['authors']) > 3 else ''}")
    else:
        print("No papers found!")


if __name__ == "__main__":
    main()
