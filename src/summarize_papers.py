"""
Summarize papers using Claude API
"""
import os
from typing import Dict, List
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class PaperSummarizer:
    """Summarizes papers using Claude API"""

    def __init__(self, api_key: str = None):
        """
        Initialize the summarizer with Claude API

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Please set it in .env file")

        self.client = Anthropic(api_key=self.api_key)

    def create_summary_prompt(self, paper: Dict) -> str:
        """
        Create a prompt for Claude to summarize the paper

        Args:
            paper: Paper information dictionary

        Returns:
            Formatted prompt string
        """
        prompt = f"""다음 논문을 한국어로 깔끔하게 요약해주세요. 전문적이면서도 읽기 쉽게 작성해주세요.

논문 제목: {paper['title']}

저자: {', '.join(paper['authors'])}

초록:
{paper['abstract']}

다음 형식으로 요약해주세요:

## 핵심 요약
(2-3문장으로 이 논문이 무엇을 다루는지 핵심만 설명)

## 주요 내용
(3-5개의 bullet point로 주요 기여와 방법론 설명)

## 의의
(이 연구가 왜 중요한지, 어떤 영향을 미칠 수 있는지 1-2문장으로 설명)

## 키워드
(관련 키워드 3-5개를 쉼표로 구분)
"""
        return prompt

    def summarize_paper(self, paper: Dict) -> str:
        """
        Summarize a single paper using Claude

        Args:
            paper: Paper information dictionary

        Returns:
            Summary text in Korean
        """
        prompt = self.create_summary_prompt(paper)

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return message.content[0].text

        except Exception as e:
            print(f"Error summarizing paper: {e}")
            return f"요약 생성 중 오류가 발생했습니다: {str(e)}"

    def summarize_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Summarize multiple papers

        Args:
            papers: List of paper information dictionaries

        Returns:
            List of papers with summaries added
        """
        summarized_papers = []

        for i, paper in enumerate(papers, 1):
            print(f"Summarizing paper {i}/{len(papers)}: {paper['title'][:60]}...")

            summary = self.summarize_paper(paper)
            paper_with_summary = {**paper, "summary": summary}
            summarized_papers.append(paper_with_summary)

        return summarized_papers


def main():
    """Test the paper summarizer"""
    # Test with a sample paper
    test_paper = {
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "et al."],
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
    }

    try:
        summarizer = PaperSummarizer()
        summary = summarizer.summarize_paper(test_paper)
        print("\n=== Test Summary ===")
        print(summary)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set ANTHROPIC_API_KEY in your .env file")


if __name__ == "__main__":
    main()
