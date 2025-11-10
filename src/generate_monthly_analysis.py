"""
Generate monthly paper analysis using Claude API
This script generates detailed analysis for monthly top 10 papers
"""
import json
import os
from datetime import datetime
from typing import List, Dict
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class MonthlyAnalysisGenerator:
    """Generates monthly paper analysis using Claude"""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        self.client = Anthropic(api_key=self.api_key)

    def create_analysis_prompt(self, papers: List[Dict], month: int, year: int = 2025) -> str:
        """Create prompt for Claude to analyze papers"""

        papers_info = ""
        for i, paper in enumerate(papers, 1):
            papers_info += f"""
## 논문 {i}
- 제목: {paper['title']}
- ArXiv ID: {paper.get('arxiv_id', 'N/A')}
- 저자: {', '.join(paper.get('authors', ['Unknown'])[:5])}
- 초록: {paper.get('abstract', 'No abstract available')}
- Upvotes: {paper.get('upvotes', 0)}

---
"""

        prompt = f"""당신은 AI/ML 분야의 최고 전문가이자 리서치 분석가입니다.
{year}년 {month}월의 Hugging Face Daily Papers에서 가장 많은 주목을 받은 Top 10 논문들을 깊이 있게 분석해주세요.

## 분석할 논문들:

{papers_info}

## 요구사항:

1. **전체 리포트 구조**:
   - 도입부에 이달의 전체 트렌드와 하이라이트 작성
   - 각 논문에 대해 아래 형식으로 상세 분석
   - 마지막에 월간 종합 분석 추가

2. **각 논문마다 다음 섹션을 포함**:

   ### 📌 TL;DR
   - 논문의 핵심을 1-2문장으로 요약

   ### 📄 기본 정보
   - 저자, 소속(파악 가능하다면), ArXiv 링크, 게재일(추정), 분야

   ### 🔬 연구 배경과 동기
   - 이 연구가 해결하려는 문제
   - 왜 이 연구가 중요한가
   - 기존 접근법의 한계

   ### 💡 핵심 아이디어
   - 제안된 방법론의 핵심 개념 (3-5개 bullet points)
   - 기존 방법과의 차별점
   - 혁신적인 부분 강조

   ### 🔧 기술적 접근
   - 사용된 주요 기술과 모델
   - 아키텍처 또는 알고리즘의 특징
   - 구현 방식의 특이사항

   ### 🌟 주요 기여점
   - 학술적 기여 (3-4개 bullet points)
   - 새로운 발견이나 개선사항

   ### 📈 실험 및 결과
   - 사용된 데이터셋이나 벤치마크
   - 정량적 성능 개선
   - 주목할 만한 발견

   ### 💪 강점과 영향력
   - 이 연구의 강점
   - 학계/산업계에 미칠 영향
   - 파급력이 큰 이유

   ### ⚠️ 한계점 및 고려사항
   - 연구의 한계
   - 추가 검증이 필요한 부분
   - 실용화 시 고려사항

   ### 🚀 응용 가능성
   - 실제 산업/제품에의 적용 가능성
   - 활용 가능한 분야

   ### 🔗 관련 연구 맥락
   - 이 논문이 속한 연구 흐름
   - 관련된 최근 연구 트렌드
   - 후속 연구 방향 제안

3. **작성 가이드라인**:
   - 전문적이면서도 읽기 쉽게
   - 단순 요약이 아닌 심층 분석 제공
   - 논문의 의미와 맥락 파악
   - 실용적인 통찰 제시
   - 체계적이고 논리적인 흐름
   - 한국어로 작성

4. **마크다운 형식**:
   - 제목과 섹션 구분 명확히
   - 이모지 적절히 사용하여 가독성 향상
   - 코드 블록, 리스트 등 적절히 활용

아래 형식으로 마크다운 문서를 생성해주세요:

```markdown
# 📚 {year}년 {month}월 AI/ML 주요 논문 Top 10 심층 분석

> {year}년 {month}월, AI 분야는 **[핵심 트렌드]**라는 거대한 전환점을 맞이했습니다. [1-2문장으로 이달의 특징 설명]
>
> 💡 **이달의 하이라이트**: [주요 발견이나 트렌드를 2-3문장으로]

---

## 목차

1. [논문 1 제목](#1-논문-1-제목-slug)
2. [논문 2 제목](#2-논문-2-제목-slug)
...
10. [논문 10 제목](#10-논문-10-제목-slug)

---

## 🏆 1. [논문 제목]

[각 논문에 대한 상세 분석...]

---

[나머지 논문들도 동일한 형식으로...]

---

## 📊 {month}월 전체 트렌드 분석

### 주요 연구 테마
1. **[테마 1]**: [설명]
2. **[테마 2]**: [설명]
3. **[테마 3]**: [설명]

### 기술적 혁신
- [혁신 1]
- [혁신 2]
- [혁신 3]

### 주목할 만한 발견
- [발견 1]
- [발견 2]

### 향후 전망
[이달의 연구들이 향후 AI 발전에 미칠 영향 예측]

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by Claude & Hugging Face*
```

위 형식을 엄격히 따라 매우 상세하고 깊이 있는 분석을 제공해주세요. 각 논문당 최소 800-1000단어 이상의 분석이 필요합니다.
"""
        return prompt

    def generate_analysis(self, papers: List[Dict], month: int, year: int = 2025) -> str:
        """Generate analysis using Claude API"""

        print(f"\n🤖 Generating analysis for {year}-{month:02d}...")
        print(f"   Analyzing {len(papers)} papers...")

        prompt = self.create_analysis_prompt(papers, month, year)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=32000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            analysis = message.content[0].text
            print(f"✅ Analysis generated successfully ({len(analysis)} characters)")
            return analysis

        except Exception as e:
            print(f"❌ Error generating analysis: {e}")
            return f"# Error\n\nFailed to generate analysis: {str(e)}"

    def save_analysis(self, analysis: str, month: int, year: int = 2025, output_dir: str = "outputs"):
        """Save analysis to markdown file"""

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/monthly_papers_{year}_{month:02d}.md"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(analysis)

        print(f"💾 Analysis saved to: {filename}")
        return filename

    def process_month(self, month: int, year: int = 2025, data_dir: str = "data", output_dir: str = "outputs"):
        """Process a single month"""

        # Load paper data
        data_file = f"{data_dir}/papers_{year}_{month:02d}.json"

        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return None

        with open(data_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        if not papers:
            print(f"❌ No papers found in {data_file}")
            return None

        print(f"\n{'='*60}")
        print(f"📚 Processing {year}-{month:02d}")
        print(f"{'='*60}")
        print(f"Found {len(papers)} papers")

        # Generate analysis
        analysis = self.generate_analysis(papers, month, year)

        # Save to file
        output_file = self.save_analysis(analysis, month, year, output_dir)

        return output_file


def main():
    """Main function"""

    generator = MonthlyAnalysisGenerator()

    # Process months 3-10
    for month in range(3, 11):
        try:
            result = generator.process_month(month, 2025)
            if result:
                print(f"✅ Successfully processed {month:02d}")
            else:
                print(f"⚠️  Skipped {month:02d} (no data)")
        except Exception as e:
            print(f"❌ Error processing {month:02d}: {e}")
            continue

    print(f"\n{'='*60}")
    print("✨ All months processed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
