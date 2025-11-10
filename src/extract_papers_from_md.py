"""
Extract paper information from markdown files
This helps reconstruct JSON data from existing analysis files
"""
import re
import json
import os
from typing import List, Dict


class MDPaperExtractor:
    """Extract paper information from markdown analysis files"""

    def extract_papers_from_md(self, md_file: str) -> List[Dict]:
        """Extract paper information from markdown file"""

        print(f"\n📖 Reading {md_file}...")

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        papers = []

        # Find all paper sections (they start with ## 🏆 N. or ## N.)
        paper_pattern = r'##\s+(?:🏆\s+)?(\d+)\.\s+(.+?)(?=\n##\s+(?:🏆\s+)?\d+\.|$)'
        paper_sections = re.findall(paper_pattern, content, re.DOTALL)

        print(f"Found {len(paper_sections)} paper sections")

        for rank, section_content in paper_sections:
            paper = self.extract_paper_info(section_content)
            paper['rank'] = int(rank)
            papers.append(paper)
            print(f"  {rank}. {paper['title'][:60]}...")

        return papers

    def extract_paper_info(self, section: str) -> Dict:
        """Extract detailed information from a paper section"""

        paper = {
            "title": "",
            "arxiv_id": "",
            "url": "",
            "huggingface_url": "",
            "authors": [],
            "abstract": "",
            "upvotes": 0,
            "published_at": "",
            "summary": ""
        }

        # Extract title (first line of section)
        title_match = re.search(r'^(.+?)(?:\n|$)', section)
        if title_match:
            paper['title'] = title_match.group(1).strip()

        # Extract ArXiv ID
        arxiv_match = re.search(r'(?:arXiv|ArXiv|arxiv)\s*(?:ID)?:?\s*(\d{4}\.\d{4,5})', section, re.IGNORECASE)
        if not arxiv_match:
            arxiv_match = re.search(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', section, re.IGNORECASE)

        if arxiv_match:
            paper['arxiv_id'] = arxiv_match.group(1)
            paper['url'] = f"https://arxiv.org/abs/{paper['arxiv_id']}"

        # Extract authors
        authors_match = re.search(r'\*\*저자\*\*:?\s*(.+?)(?:\n|$)', section, re.IGNORECASE)
        if authors_match:
            authors_text = authors_match.group(1)
            # Clean up authors
            authors_text = re.sub(r'\(.*?\)', '', authors_text)  # Remove parentheses
            authors_text = re.sub(r'\d+명.*?저자', '', authors_text)  # Remove "N명의 공동 저자"
            authors = [a.strip() for a in re.split(r'[,،]', authors_text) if a.strip()]
            paper['authors'] = [a for a in authors if len(a) > 2 and not a.startswith('외')][:10]

        # Extract upvotes
        upvotes_match = re.search(r'(?:Upvotes?|추천):?\s*(\d+)', section, re.IGNORECASE)
        if upvotes_match:
            paper['upvotes'] = int(upvotes_match.group(1))

        # Extract published date
        date_match = re.search(r'(?:발표일|게재일|Published):?\s*(\d{4})년?\s*(\d{1,2})월?\s*(\d{1,2})일?', section)
        if date_match:
            year, month, day = date_match.groups()
            paper['published_at'] = f"{year}-{int(month):02d}-{int(day):02d}"

        # Extract TL;DR as summary
        tldr_match = re.search(r'>\s*\*\*TL;DR\*\*:?\s*(.+?)(?:\n\n|###)', section, re.DOTALL)
        if tldr_match:
            paper['summary'] = tldr_match.group(1).strip()

        # Extract abstract (연구 배경 section as proxy)
        abstract_match = re.search(r'###\s+🎯\s+연구\s+배경.*?\n\n(.+?)(?=\n###)', section, re.DOTALL)
        if abstract_match:
            paper['abstract'] = abstract_match.group(1).strip()[:500]

        return paper

    def save_to_json(self, papers: List[Dict], output_file: str):
        """Save papers to JSON file"""

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(papers)} papers to {output_file}")


def main():
    """Main function to extract papers from existing MD files"""

    extractor = MDPaperExtractor()

    # Extract from January and February files
    for month in [1, 2]:
        md_file = f"outputs/monthly_papers_2025_{month:02d}.md"
        json_file = f"data/papers_2025_{month:02d}.json"

        if not os.path.exists(md_file):
            print(f"⚠️  {md_file} not found, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {md_file}")
        print(f"{'='*60}")

        try:
            papers = extractor.extract_papers_from_md(md_file)
            extractor.save_to_json(papers, json_file)
            print(f"✅ Successfully extracted data for 2025-{month:02d}")
        except Exception as e:
            print(f"❌ Error processing {md_file}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("✨ Extraction complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
