---
title: "Delta Lake Bloom Filter: 적용보다 제거가 나았던 이유"
subtitle: "성능 기능은 적용보다 재검증과 제거 기준이 중요하다"
tags: [Delta Lake, Databricks, Photon, Optimization]
comments: true
---

> 이 글의 수치와 사례는 특정 회사·서비스·데이터를 식별할 수 없도록 일반화했습니다.

## 요약

대용량 Delta 테이블 조회를 빠르게 만들어보려고 Bloom Filter를 적용했다가, 나중에 다시 제거한 이야기입니다. 작년에는 1~3건 point lookup에서 제한적인 성능 향상을 확인해 그 범위에 적용했습니다. 다만 다건 조회와 조인에서는 개선 폭이 작았고, 비교 시점마다 테이블의 데이터량·파일 구성·정렬 상태도 달랐습니다. 이후에도 계속 확신이 서지 않아 올해 초 같은 조건에서 다시 점검했습니다. 작년에는 Bloom Filter 유무와 관계없이 Photon 사용 양상이 같아 문제가 되지 않았지만, 올해 재검증에서는 Bloom Filter가 Photon 경로를 막고 있었습니다. 한 달 뒤 공식 문서를 다시 확인하니 Bloom Filter는 deprecated 처리되어 Predictive I/O로 대체된 상태였습니다. 이론적으로 나아질 것 같은 기능도 적용 시점의 조건과 엔진 변화를 함께 보며 재검증해야 한다는 것을 다시 확인한 경험이었습니다.

## 1. Bloom Filter를 왜 붙였었나

대용량 로그 테이블을 다루다 보면, 특정 사용자나 기기 식별자로 찾아야 하는 조회가 종종 느리게 나옵니다. 날짜 파티션만으로는 이런 조회에서 파일 스킵이 잘 안 되는 경우가 있어서, 뭔가 더 빠르게 만들 방법이 없을까 찾아보다가 Bloom Filter를 알게 됐습니다.

Bloom Filter는 행을 직접 줄여주는 기능은 아니고, "이 파일엔 그 값이 없을 가능성이 높다"를 미리 판단해서 **읽기 전에 파일을 걸러내는** 보조 수단입니다. 이론적으로는 고카디널리티 컬럼을 조회할 때 딱 맞는 기능처럼 보였습니다.

## 2. 그때 비교했던 선택지

| 선택지 | 기대 효과 | 비용·주의점 |
|---|---|---|
| 기본 통계·파티션 | 단순하고 유지비 낮은 file pruning | 고카디널리티 조건에서는 한계 |
| Bloom Filter | 소수 키 point lookup의 파일 스킵 보완 | 인덱스 생성·유지와 계획 비용 |
| Z-ORDER | 관련 값을 파일 단위로 모아 skipping 강화 | 재정렬 비용이 높고 쓰기 workload에 영향 |
| REORG | Deletion Vector 물질화·파일 정리 | 실행 시 읽기·쓰기 비용 발생 |
| Photon·Predictive I/O | 엔진이 조건과 파일을 더 효율적으로 판단 | 런타임·테이블 상태에 따라 재검증 필요 |

## 3. 작년엔 이렇게 판단했습니다

작년에는 1~3건의 소수 키로 point lookup을 하는 워크로드를 세 차례 반복 측정했습니다. 단일 키 조회에서는 실행시간이 줄어드는 관측이 있었고, 그 범위에서 효과가 있다고 판단해 운영에 적용했습니다. 반면 다건 조회와 조인에서는 개선 폭이 미미해 적용 범위를 넓히지 않았습니다.

다만 당시 비교한 테이블은 시점별로 데이터량·파일 수·정렬 상태가 달랐고, 일부 비교에는 Z-ORDER도 함께 적용되어 있었습니다. 따라서 그 결과만으로 Bloom Filter 자체의 효과를 일반화할 수는 없었습니다. 작년에는 Bloom Filter 유무와 관계없이 Photon 사용 양상도 같았습니다. 당시에는 Predictive I/O 최적화 경로가 거의 활용되지 않아 Bloom Filter가 막을 대상이 없었던 것입니다. 그사이 Photon을 비롯한 여러 최적화 기능이 계속 바뀌고 있었기 때문에, 지금도 같은 판단이 유효한지 확인하기로 했습니다.

## 4. 확신이 서지 않아 다시 확인해봤습니다

### 4.1 이번엔 워크로드를 나눠서 봤습니다

작년에도 단일·다건 조회와 조인을 나눠 확인했지만, 비교 시점의 테이블 상태가 같지는 않았습니다. 또한 최적화 방식에 따라 프루닝 비율의 분모가 되는 후보 파일 집합이 달라질 수 있어, 프루닝 비율만으로 기능 효과를 비교하기 어렵습니다. 이번에는 실제로 매일 실행되는 **컴플라이언스 삭제 워크로드**와 **특정 식별자 조회 워크로드**를 나눠, 같은 테이블·같은 조건에서 Bloom Filter가 있을 때와 없을 때를 직접 비교했습니다. 절대적인 파일·바이트 읽기량, 실행계획, Photon 사용 비중을 함께 확인했습니다.

### 4.2 삭제 워크로드: 효과가 없었습니다

컴플라이언스 삭제는 특정 식별자를 기준으로 매일 반복되는 작업이었습니다. Bloom Filter 유무에 따라 삭제 실행계획의 스캔 지표를 뽑아봤는데, 아래가 그대로 반복해서 나왔습니다.

- **number of bytes before skipping = number of bytes after skipping**
- **number of files before skipping = number of files after skipping**
- 전체 실행시간의 대부분은 파일을 다시 쓰는 시간이 아니라 **삭제할 대상을 찾느라 스캔하는 시간**이었습니다 (스캔은 50분대, 재작성은 5초도 안 걸림)
- 실제로 물리적으로 지워진 행은 0건 — 삭제는 Deletion Vector로만 반영됨

<!-- 스크린샷 자리 ①: EXPLAIN FORMATTED DELETE 실행계획에서 bytes/files before·after skipping이 동일하게 나온 부분 (테이블·스키마명은 캡처 전에 가려주세요) -->

Bloom Filter가 있어도 삭제 워크로드의 파일·바이트 pruning에는 아무 차이가 없었습니다. 삭제가 느린 건 애초에 "대상이 어느 파티션에 있는지 미리 좁히기 어려운" scan-heavy한 구조 자체가 원인이었고, Bloom Filter를 뺀다고 더 빨라지지도 않았습니다 — **삭제 쪽에서는 있어도 없어도 그냥 중립**이었습니다.

### 4.3 조회 워크로드: Bloom Filter를 제거한 편이 나았습니다

식별자 하나 또는 몇 개로 조회하는 케이스도 다시 봤습니다. 같은 조건으로 Bloom Filter를 생성 → 제거 → 재생성 → 제거(캐시 반영 후) 순서로 반복 조회하면서 실행계획과 실행시간을 비교했습니다.

실행계획부터 눈에 띄게 달랐습니다.

- Bloom Filter 적용 시: `Scan parquet with Bloom Filters ...` 경로를 타면서, Photon explanation에 `Photon scan does not support data sources with bloom filter indexes`가 그대로 찍혔습니다.
- Bloom Filter 제거 시: `PhotonScan parquet ...`로 바뀌고, 조건 필터는 똑같이 반영됐습니다.

<!-- 스크린샷 자리 ②: Bloom Filter 적용 시/제거 시 EXPLAIN 결과 나란히 비교 (Photon scan does not support... 문구가 보이는 부분) -->

실측 결과도 같은 방향을 가리켰습니다.

- Files read / Files pruned / Partitions read / Bytes pruned는 Bloom Filter 유무와 **동일** — 추가로 걸러지는 게 없었습니다.
- Photon 사용 비중: 적용 시 1~2% → 제거 시 99%
- Rows read: 적용 시 수억 건대 → 제거 시 수천 건대로 뚝 떨어짐
- 실행시간: 단건 조회 약 1분25초 → 약 45초, 소건(3개) 조회 약 1분10초 → 약 30초대로 개선

<!-- 스크린샷 자리 ③: Query History에서 Bloom Filter 적용/제거 실행시간·Photon 사용률 비교 그래프 -->

작년의 제한적인 성능 향상 관측이 틀렸다는 뜻은 아니었습니다. 다만 당시에는 Bloom Filter 유무와 관계없이 Photon 사용 양상이 같았고, Predictive I/O 최적화 경로가 거의 활용되지 않았습니다. Bloom Filter가 Photon의 이점을 가로막는 문제는 그때는 없었던 것입니다.

올해 같은 조건에서 다시 보니 Bloom Filter가 파일을 추가로 걸러주는 효과는 크지 않았고, 오히려 Photon 경로를 막고 있다는 점이 드러났습니다. 플랫폼의 Predictive I/O 최적화가 그 사이 발전하면서, 작년에는 무해했던 선택이 올해는 그 발전을 막는 선택이 된 것입니다. 지금 조건에서는 Bloom Filter가 얻는 것보다 잃는 게 크다고 판단했고, Bloom Filter를 제거한 뒤 Z-ORDER 적용 범위를 다시 검토했습니다.

### 4.4 공식 문서에서 답을 확인했습니다

재검증 한 달 뒤 Databricks 공식 문서를 다시 봤는데, Bloom Filter 인덱스가 이미 deprecated 처리돼 있었습니다. 대체 수단으로는 Predictive I/O가 명시돼 있었습니다.

> "Databricks has deprecated this feature and recommends removing any existing Bloom filter indexes from your tables."
>
> "Predictive I/O performs file skipping on all columns automatically. It fully supersedes Bloom filter indexes, which only add write overhead when Photon is enabled."
>
> — [Databricks: Bloom filter index deprecation](https://docs.databricks.com/aws/en/optimizations/bloom-filters)

운영 테이블에서 직접 확인한 것과 벤더가 공식적으로 권고하는 방향이 같았습니다. Predictive I/O가 Bloom Filter의 file skipping 역할을 자동으로, 그리고 Photon 경로를 막지 않으면서 대체해준다는 뜻이었습니다.

## 5. Z-Order / Liquid Clustering은 지금 어떻게 하고 있나

Bloom Filter를 떼어낸 다음 자연스럽게 Z-ORDER를 다시 봤는데, 막상 적용 범위를 생각해보니 다른 고민이 있었습니다.

- **Z-Order**: 특정 조건 조회는 확실히 개선될 여지가 있지만, 대규모 append + delete가 계속 도는 운영 테이블 전체에 걸기엔 재정렬 비용이 만만치 않았습니다. 그래서 지금은 hot range나 조회 빈도가 높은 일부 워크로드에 한정해서 검토하고 있습니다.
- **Liquid Clustering**: 기존 파티션 구조·운영 방식과 바로 맞지 않고, 대용량 데이터를 재정렬하는 비용이 커서 아직 표준 전략으로 채택하지는 않았습니다.

## 6. 느낀 점

- 작년의 제한적인 성능 향상 관측은 유효했지만, 비교 조건과 워크로드 범위를 벗어나 일반화할 수는 없었습니다.
- 기능을 한 번 적용하고 끝내는 게 아니라, workload나 런타임이 바뀔 때마다 다시 검증해야 한다는 걸 몸으로 배웠습니다.
- 이 과정에서 Parquet·Delta 내부 동작(파일 스킵, 실행계획, Photon 경로)을 꽤 깊게 파볼 수 있었던 것도 나름 소득이었습니다.

## 마무리

Bloom Filter를 적용했다가 다시 제거한 이야기지만, 결국 남는 교훈은 그때의 조건에서 맞았던 결정도 엔진과 workload가 바뀌면 다시 확인해야 한다는 것입니다. 지금은 Predictive I/O가 어떻게 Bloom Filter 없이도 이 정도의 file skipping을 자동으로 해내는지가 궁금해졌습니다. 기회가 되면 다음 글에서 이 부분을 다뤄보려고 합니다.
