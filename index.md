---
layout: single
title: "Data Platform Engineer"
excerpt: "데이터가 안정적으로 흐르고, 다시 측정되며, 오래 운영될 수 있도록 설계합니다."
author_profile: true
classes: wide
---

## 데이터 플랫폼을 운영 가능한 시스템으로 만듭니다

데이터 플랫폼·파이프라인·Lakehouse를 설계하고 운영합니다. **Delta Lake 최적화**, 증분 적재, 데이터 품질과 관측성처럼 기능 자체보다 workload와 운영 비용을 함께 판단하는 일을 좋아합니다.

### Core areas

- **Lakehouse** — Delta Lake, Spark, Databricks, 파일·테이블 레이아웃 최적화
- **Data pipelines** — 증분 적재, MERGE, streaming, 재처리와 실패 복구
- **Platform operations** — 권한, 비용, 관측성, 문서화와 셀프서비스

## Latest notes

{% assign public_posts = site.posts | where_exp: "post", "post.legacy != true" %}
{% if public_posts.size > 0 %}
{% for post in public_posts limit: 10 %}
<article class="archive__item">
  <h3 class="archive__item-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
  <p class="page__meta">{{ post.date | date: "%Y.%m.%d" }}</p>
  {% if post.excerpt %}<p class="archive__item-excerpt">{{ post.excerpt | strip_html | strip_newlines | truncate: 180 | escape }}</p>{% endif %}
</article>

{% endfor %}
{% else %}
아직 공개 승인된 게시물이 없습니다. 검토 중인 글은 초안으로 보존하고, 공개가 확정된 글부터 이 목록에 추가합니다.
{% endif %}
