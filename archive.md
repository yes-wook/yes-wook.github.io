---
layout: single
title: "Archive"
permalink: /archive/
author_profile: false
classes: wide
---

2018년 이전에 작성한 글은 새 글 목록과 분리해 보관합니다. 기존 게시물의 permalink는 유지하고, 이 페이지에서만 다시 찾아볼 수 있도록 운영합니다.

{% assign legacy_posts = site.posts | where: "legacy", true %}
{% if legacy_posts.size > 0 %}
{% for post in legacy_posts %}
<article class="archive__item">
  <h3 class="archive__item-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
  <p class="page__meta">{{ post.date | date: "%Y.%m.%d" }}</p>
  {% if post.excerpt %}<p class="archive__item-excerpt">{{ post.excerpt | strip_html | strip_newlines | truncate: 180 | escape }}</p>{% endif %}
</article>

{% endfor %}
{% else %}
아직 이 저장소에 보존된 이전 게시물 원본이 없습니다. 원본을 추가하면 `legacy: true`와 기존 permalink를 유지한 채 이 목록에 표시됩니다.
{% endif %}
