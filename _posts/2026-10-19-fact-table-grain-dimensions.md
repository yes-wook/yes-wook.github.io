---
layout: post
title: "팩트테이블은 집계 테이블이 아니다 — grain부터 정하기"
subtitle: "주문 한 건과 주문상품 한 건의 차이에서 시작하는 Fact와 Dimension"
description: "주문·물류 예제로 grain과 Fact/Dimension, 팩트의 세 유형, 가산성, 상세 팩트와 집계 테이블의 관계를 정리합니다."
date: 2026-10-19 00:00:00 +0900
categories: ["개념 노트"]
tags: [Data Modeling, Data Warehouse, Fact, Dimension]
comments: true
---

**팩트테이블은 집계 결과만을 뜻하지 않습니다. 먼저 한 행이 무엇을 나타내는지, grain을 정해야 합니다.**

팩트테이블을 설명하려고 하면 “사실을 담는 테이블”, “집계에 사용하는 테이블”이라는 말이 먼저 떠오릅니다. 하지만 이 설명만으로는 한 행에 무엇을 담아야 하는지 정하기 어렵습니다.

예를 들어 주문 테이블에서는 행 수를 세면 주문 수가 됩니다. 여기에 주문상품을 조인하면 어떻게 될까요? 한 주문에 상품이 여러 개 있으면 행이 늘어납니다. 같은 주문번호가 보이더라도 `COUNT(*)`가 의미하는 값은 달라집니다.

이 차이를 설명하는 출발점이 **grain**, 즉 한 행이 나타내는 업무 단위입니다. 이 글에서는 가상의 주문·물류 데이터를 사용해 grain, Fact, Dimension을 정리하고, 상세 팩트와 집계 테이블이 어떻게 연결되는지 살펴보겠습니다.

## 1) 주문은 두 건인데, 왜 세 건으로 집계될까?

다음은 주문 한 건을 한 행으로 저장한 가상 데이터입니다. 금액 단위는 원입니다.

| 주문번호 | 주문금액 |
|---|---:|
| O-001 | 30,000 |
| O-002 | 20,000 |

주문은 두 건이고 총 주문금액은 50,000원입니다. 여기에 주문상품을 연결하면 다음과 같은 결과가 나올 수 있습니다.

| 주문번호 | 주문상품번호 | 상품 | 상품금액 | 주문금액 |
|---|---|---|---:|---:|
| O-001 | 1 | 사과 | 10,000 | 30,000 |
| O-001 | 2 | 배 | 20,000 | 30,000 |
| O-002 | 1 | 사과 | 20,000 | 20,000 |

이 결과에서 `COUNT(*)`는 3이고, `SUM(주문금액)`은 80,000원입니다. 주문 수와 주문금액으로 해석하면 둘 다 잘못된 값입니다. 한 행은 이제 주문 한 건이 아니라 **주문상품 한 건**을 나타내고, 주문 전체 금액이 상품 행마다 반복되기 때문입니다.

`COUNT(DISTINCT 주문번호)`로 이 결과의 주문 수를 구할 수는 있습니다. 그러나 그것만으로 반복된 주문금액까지 해결되지는 않습니다. `SUM(DISTINCT 주문금액)`도 해법이 아닙니다. 금액이 같은 서로 다른 주문까지 하나로 계산하기 때문입니다.

이 예시에서는 주문 단위 금액을 주문 팩트에 두고, 주문상품 팩트에는 상품별 금액을 둡니다. 상품별로 주문 전체 금액을 분석해야 한다면 별도의 배분 규칙이 필요합니다. 배송비나 주문 단위 할인처럼 상품에 바로 귀속되지 않는 금액도 같은 판단이 필요합니다.

## 2) Grain은 키 목록보다 먼저 정하는 행의 의미입니다

Grain은 팩트테이블의 한 행이 무엇을 나타내는지에 대한 정의입니다. Kimball은 업무 프로세스를 선택한 뒤 grain을 선언하고, 그다음 차원과 측정값을 정하는 순서를 제시합니다. 차원과 측정값이 모두 같은 행 단위에 맞아야 하기 때문입니다. [Grain](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/grain/), [Four-Step Dimensional Design Process](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/four-4-step-design-process/)

가상의 물류 데이터를 설계한다면 다음과 같이 선언할 수 있습니다.

| 테이블 | 한 행의 의미 | 예시 식별 기준 |
|---|---|---|
| 주문 팩트 | 접수된 주문 한 건 | 주문번호 |
| 주문상품 팩트 | 주문 내 상품 라인 한 건 | 주문번호 + 주문상품번호 |
| 피킹 완료 이벤트 팩트 | 피킹 완료 이벤트 한 건 | 이벤트 ID |
| 시간별 처리량 집계 팩트 | 센터 한 곳의 한 시간 처리량 | 센터 키 + 시간 |

“주문번호와 상품번호를 키로 사용한다”는 설명만으로 충분하지 않을 수도 있습니다. 같은 상품이 한 주문에 여러 라인으로 들어올 수 있는지, 한 라인이 여러 작업으로 나뉘는지에 따라 실제 식별 기준이 달라집니다.

따라서 먼저 “한 행은 무엇인가”를 문장으로 적고, 그 정의를 만족하는 키와 중복 검증 규칙을 정하는 편이 좋습니다. 주문·주문상품·피킹 작업을 연결할 때도 각 테이블 사이의 일대다 관계부터 확인해야 합니다.

컬럼을 하나 추가한다고 grain이 자동으로 바뀌는 것은 아닙니다. 주문당 한 행에 현재 상태를 갱신하면 여전히 주문 grain입니다. 반면 상태가 바뀔 때마다 새 행을 쌓으면 한 행은 상태 변경 이벤트가 됩니다. 이 경우 같은 주문이 여러 번 등장하므로 `COUNT(*)`는 주문 수가 아니라 이벤트 수가 됩니다.

## 3) Fact는 측정을, Dimension은 해석할 맥락을 담습니다

팩트테이블에는 업무에서 발생한 측정 사건과 그에 맞는 측정값을 담습니다. 가장 상세한 수준에서는 개별 사건이 한 행이 될 수 있습니다. 관련 차원을 연결할 키도 함께 둡니다. 팩트가 처음부터 집계된 결과여야 하는 것은 아닙니다. [Fact Table Structure](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/fact-table-structure/)

피킹 완료 이벤트를 예로 들면 처리 수량은 측정값이고, 처리한 센터·상품·날짜는 분석의 맥락입니다. 센터 이름이나 상품 카테고리처럼 필터와 그룹화에 사용할 설명 속성은 디멘션테이블에 둡니다. [Dimension Table Structure](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/dimension-table-structure/)

```text
dim_center ──┐
dim_product ─┼── fact_pick_event
dim_date ────┘     한 행 = 피킹 완료 이벤트 한 건
                   측정값 = 처리 수량
```

예를 들어 `dim_product`에 상품명과 카테고리명을 함께 두면 “카테고리별 처리 수량”을 조회할 수 있습니다. 차원모델에서는 조회의 단순성과 속도를 위해 계층 속성을 한 디멘션 행에 펼쳐 두기도 합니다. 따라서 앞선 주문금액 중복 문제를 단순히 “정규화가 안 됐다”는 말로 설명하기는 어렵습니다. 측정값의 grain과 조인 관계를 확인해야 합니다. [Denormalized Flattened Dimensions](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/denormalized-flattened-dimension/)

Fact와 Dimension을 append와 update로 구분하는 것도 충분하지 않습니다. 디멘션의 속성 변경은 기존 값을 덮어쓸 수도 있고(SCD Type 1), 새로운 버전 행을 추가해 이력을 남길 수도 있습니다(SCD Type 2). 아래에서 볼 팩트 유형 중에도 기존 행을 갱신하는 경우가 있습니다. [Type 1](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/type-1/), [Type 2](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/type-2/)

가령 서울에 살던 고객이 부산으로 이사했다고 가정하겠습니다. 현재 지역만 부산으로 덮어쓰고 과거 주문을 그 값에 연결하면 과거 주문도 부산으로 분류됩니다. 주문 당시 지역을 분석하려면 당시 정보를 보존하고 연결해야 합니다. Type 2에서는 고객 정보의 버전마다 별도 키와 유효 기간을 두고, 팩트를 해당 시점의 버전에 연결할 수 있습니다. 고객 ID만으로 모든 버전과 조인하면 주문 행이 늘어날 수 있으므로 버전 키나 유효 기간 조건이 필요합니다.

## 4) 팩트는 사건·기간·프로세스를 표현할 수 있습니다

팩트테이블의 대표적인 세 유형은 다음과 같습니다.

| 유형 | 한 행의 의미 | 가상 예시 |
|---|---|---|
| Transaction fact | 특정 시점의 측정 사건 | 피킹 완료 이벤트 한 건 |
| Periodic snapshot fact | 정해진 기간의 활동이나 상태 | 센터·상품별 하루의 입출고 수량과 마감 재고 |
| Accumulating snapshot fact | 시작과 종료, 주요 단계가 정의된 프로세스 한 단위 | 출고 작업 한 건의 접수·피킹·포장·출고 진행 |

Transaction fact는 측정 사건이 발생했을 때 기록합니다. Periodic snapshot은 하루나 한 달처럼 정해진 기간을 기준으로 기록하며, 활동이 없는 기간에도 0 또는 null을 가진 행을 두는 구성이 일반적입니다. [Transaction Fact Tables](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/transaction-fact-table/), [Periodic Snapshot Fact Tables](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/periodic-snapshot-fact-table/)

Accumulating snapshot은 프로세스가 진행되면서 같은 행의 단계별 시점 등을 갱신합니다. 가령 출고 작업을 접수했을 때 행을 만들고, 피킹·포장·출고가 완료될 때 각 완료시각을 채우는 방식입니다. 주문이 여러 출고 작업으로 분할될 수 있다면 이 예시의 grain은 주문이 아니라 출고 작업으로 잡아야 합니다. [Accumulating Snapshot Fact Tables](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/accumulating-snapshot-fact-table/)

이 세 유형은 업무를 시간에 따라 어떻게 표현할지에 대한 선택입니다. 상세 데이터를 더 높은 단위로 요약한 aggregate fact는 집계 수준에 관한 개념이므로, 이 세 유형과 단순히 같은 기준의 네 번째 항목으로 나열하면 혼동하기 쉽습니다.

## 5) 숫자라고 모두 더할 수 있는 것은 아닙니다

측정값을 설계할 때는 어떤 축으로 합산할 수 있는지도 정해야 합니다. Kimball은 이를 가산·준가산·비가산으로 구분합니다. [Additive, Semi-Additive, and Non-Additive Facts](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/additive-semi-additive-non-additive-fact/)

| 구분 | 의미 | 예시와 조건 |
|---|---|---|
| 가산 | 관련된 모든 차원으로 합산 가능 | 중복 없이 기록된 동일 단위 상품의 처리 수량 |
| 준가산 | 일부 차원으로만 합산 가능 | 같은 시점·같은 상품의 센터별 재고 수량 |
| 비가산 | 값 자체를 더해 재집계하기 어려움 | 평균 처리시간, 완료율 |

예를 들어 같은 상품의 마감 재고가 월요일 100개, 화요일 120개라면 두 값을 더한 220개를 화요일 재고라고 할 수 없습니다. 반면 화요일의 A센터 재고 100개와 B센터 재고 120개는 재고 범위가 겹치지 않고 기준 시점이 같다면 220개로 합산할 수 있습니다.

평균도 주의해야 합니다. A센터가 2건을 처리하는 데 총 20분, B센터가 8건을 처리하는 데 총 160분이 걸렸다고 가정하겠습니다.

| 센터 | 처리 건수 | 총 처리시간 | 평균 처리시간 |
|---|---:|---:|---:|
| A | 2 | 20분 | 10분 |
| B | 8 | 160분 | 20분 |

센터별 평균을 단순 평균하면 15분이지만, 전체 10건의 평균은 `(20 + 160) / (2 + 8) = 18분`입니다. 이 예시의 처리시간은 각 완료 건의 소요시간을 합한 값이며, 두 센터에서 측정 정의가 같다고 가정합니다.

따라서 평균값만 저장하기보다 총 처리시간과 처리 건수를 함께 보존하면 재집계가 가능합니다. 완료율도 같은 방식으로 완료 건수와 대상 건수를 보존한 뒤 최종 조회 단위에서 나눕니다.

주문 수도 조회 기준에 주의해야 합니다. 상품별 주문 수는 한 주문이 여러 상품에 포함될 수 있으므로 상품별 값을 더한 결과가 전체 주문 수와 같지는 않습니다.

## 6) 상세 팩트와 집계 테이블을 함께 설계합니다

“센터별 시간대별 처리 수량” 대시보드를 만든다면 미리 계산한 집계 테이블이 유용할 수 있습니다. 반복되는 조회에서 읽어야 할 데이터 양을 줄일 수 있기 때문입니다. Kimball도 aggregate fact를 상세 팩트의 조회 성능을 높이기 위한 요약 테이블로 설명합니다. [Aggregate Fact Tables or Cubes](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/aggregate-fact-table-cube/)

하지만 센터와 시간만 남기고 집계하면 새로운 질문을 다루기 어려워집니다. “특정 상품의 처리량만 보고 싶다”, “어느 이벤트 때문에 수량이 늘었는지 확인하고 싶다”는 요구에는 집계 과정에서 없어진 정보가 필요합니다.

```text
상세 팩트: 피킹 완료 이벤트 한 건
    │
    └─ 센터·시간별 GROUP BY
           ↓
집계 팩트: 센터 한 곳의 한 시간 처리량
```

아래는 가상의 상세 팩트에서 시간별 집계 팩트를 만드는 Spark SQL 예시입니다. 완료 이벤트만 들어 있고, `processed_qty`의 단위가 같으며, 이벤트 중복이 제거되어 있다고 가정합니다. 시간대 기준도 미리 통일해야 합니다.

```sql
SELECT
    center_key,
    date_trunc('HOUR', completed_at) AS completed_hour,
    SUM(processed_qty) AS processed_qty
FROM fact_pick_event
GROUP BY
    center_key,
    date_trunc('HOUR', completed_at);
```

이 결과의 한 행은 센터 하나의 한 시간 처리량입니다. 상품별 분석이 필요하면 상품 키를 포함한 다른 grain으로 집계하거나 상세 팩트를 조회해야 합니다. 이 쿼리만으로는 이벤트가 없는 시간의 0행이 생성되지도 않습니다. 빈 시간까지 보여줘야 한다면 센터·시간 기준 집합과 연결하는 처리가 추가로 필요합니다.

상세 팩트는 새로운 필터와 원인 추적에 대응할 수 있는 기반이고, 집계 팩트는 반복되는 질문에 빠르게 답하기 위한 수단입니다. 어떤 집계를 만들지는 실제 조회 패턴과 성능 요구를 확인해 결정합니다.

## 7) 테이블을 만들기 전에 확인할 질문

이 내용을 설계에 적용한다면 다음 질문부터 적어보겠습니다.

1. 한 행은 어떤 사건·기간·프로세스를 나타내는가?
2. 그 단위를 유일하게 식별하는 기준은 무엇인가?
3. 측정값이 모두 그 단위에 맞는가? 상위 단위 금액이 반복되어 있지는 않은가?
4. 각 디멘션을 조인한 뒤에도 팩트 행 수가 유지되는가? 이력형 디멘션은 올바른 버전과 연결되는가?
5. 측정값은 어떤 차원으로 더할 수 있는가? 재고·평균·비율의 재집계 규칙은 무엇인가?
6. 집계 과정에서 어떤 분석 기준이 사라지는가? 상세 데이터로 돌아갈 경로가 있는가?

팩트테이블은 상세 사건도, 기간별 상태도, 프로세스의 진행도 담을 수 있습니다. 집계 팩트 역시 팩트테이블의 한 형태입니다. 제목의 “집계 테이블이 아니다”는 **팩트테이블을 집계 결과만으로 정의할 수 없다는 의미**입니다.

설계의 출발점은 한 행의 의미를 정하는 일입니다. 그 정의가 분명해야 주문 수, 상품 수, 처리 수량처럼 비슷해 보이는 숫자를 올바르게 계산할 수 있습니다.
