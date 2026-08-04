꾸미는 것이 아니라 정보를 디자인하는 것입니다.

Markdown은 제약이 있기 때문에, 오히려 규칙을 정해두면 훨씬 전문적으로 보입니다.

⸻

APC Markdown Style Guide

1. 제목은 최대 H3까지만

# Wire Saw APC
## 1. 데이터
### 1.1 개요

❌ H4, H5 남발하지 않기

⸻

2. 핵심은 항상 회색 인용문

Markdown에서 가장 예쁜 요소입니다.

> Leakage 제거가 가장 큰 성능 향상을 가져왔다.

↓

Leakage 제거가 가장 큰 성능 향상을 가져왔다.

논문의 Abstract처럼 사용합니다.

⸻

3. 강조는 Bold 하나만

**Leakage 제거**

굵게만 사용합니다.

❌

***굵고 기울임***
<u>밑줄</u>
<font color>

사용하지 않습니다.

⸻

4. 표는 최대한 심플하게

좋은 예

| 항목 | 값 |
|------|----|
| Target | BOW |
| Rows | 42,000 |
| Model | XGBoost |

나쁜 예

| 📊항목 | 📈값 |
|⭐⭐⭐⭐⭐|

⸻

5. 리스트는 5개 이하

- 데이터
- 모델
- 결과

7~8개 이상이면 표로 변경합니다.

⸻

6. 구분선 적극 사용

---
## 모델
---
## 결과

문단이 훨씬 읽기 편합니다.

⸻

7. 코드블록은 언어 지정

```python
model.fit(X, y)
```

GitLab Syntax Highlighting이 적용됩니다.

⸻

8. 이미지는 항상 캡션 추가

Figure 1. Feature Importance

또는

> Figure 1. Leakage 제거 후 Feature Importance

⸻

9. 중요한 숫자는 표

예를 들어

❌

MAE는 0.18이고
RMSE는 0.24이고...

대신

| Metric | Value |
|---------|------:|
| MAE | 0.18 |
| RMSE | 0.24 |

⸻

10. 긴 문장은 쓰지 않기

가능하면

한 문장 = 한 줄

예

Leakage 제거 후 데이터를 다시 생성하였다.
총 데이터는 37,340건이다.
성능이 크게 향상되었다.

GitLab에서 읽기가 매우 편합니다.

⸻

제가 가장 추천하는 Highlight

Markdown에서 제일 예쁜 건 사실 이것 하나입니다.

> Insight
>
> Leakage 제거보다 중요한 Feature는 발견되지 않았다.

↓

Insight

Leakage 제거보다 중요한 Feature는 발견되지 않았다.

문서 전체에서 통일해서 사용하면 상당히 깔끔합니다.

⸻

아이콘도 최소한만

저는 아래 정도만 씁니다.

아이콘	용도
📌	중요한 항목
💡	Insight
⚠️	주의사항
✅	완료
📊	결과
🔍	분석

그 이상은 문서가 산만해질 가능성이 큽니다.

⸻

APC 문서에서 가장 추천하는 구성

예를 들어 모델결과.md라면

# 모델 결과
> 최종 모델 성능 및 검증 결과
---
## 성능 요약
| Metric | Value |
|---------|------:|
| MAE | 0.18 |
| RMSE | 0.24 |
> **결론**
>
> XGBoost가 가장 높은 성능을 보였다.
---
## Feature Importance
(이미지)
> **Insight**
>
> Equipment ID 의존도가 높다.
---
## Error Analysis
(표)
> **Action**
>
> Leakage 제거 후 재실험 필요.

⸻

프로젝트에서 꼭 넣고 싶은 규칙 하나

각 문서의 마지막에 “Summary” 섹션을 고정으로 두는 것입니다.

---
## Summary
### Key Points
- Leakage 제거가 가장 큰 성능 향상을 보였다.
- XGBoost가 최고 성능을 기록했다.
- 일반화 성능 확보가 다음 과제이다.
### Next Action
- CatBoost 재실험
- SHAP 분석
- Recipe 추천 검증

이렇게 하면 긴 문서를 다시 읽지 않아도 핵심 내용과 다음 할 일을 마지막에서 바로 확인할 수 있습니다.

⸻

이 스타일의 핵심은 꾸미기보다 일관성입니다. 제목 계층, 인용문, 표, 구분선, 요약 형식만 모든 .md 파일에서 동일하게 유지해도 GitLab에서는 매우 깔끔하고 전문적인 문서로 보입니다.
