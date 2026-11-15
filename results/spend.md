# API spend

| time | run | what | USD | running total |
|---|---|---|---|---|
| 2026-11-14T22:05:36 | 2026-11-14_hr_baseline | index + search | 0.0188 | 0.0188 |
| 2026-11-14T22:06:29 | 2026-11-14_hr_baseline | answers (20 queries) | 0.0242 | 0.0430 |
| 2026-11-14T22:09:40 | 2026-11-14_hr_baseline | answers (318 queries) | 0.3407 | 0.3837 |
| 2026-11-14T22:40:17 | 2026-11-14_hr_pilot_rerank_llm | index + search | 0.0143 | 0.3979 |
| 2026-11-14T22:51:16 | pilot_contextual_headers | 21 pages | 0.0051 | 0.4030 |
| 2026-11-14T23:02:55 | 2026-11-14_hr_s4_rerank_llm | index + search | 0.2145 | 0.6175 |
| 2026-11-14T23:42:10 | 2026-11-14_hr_x_ctx_no_rerank | index + search | 0.5968 | 1.2143 |
| 2026-11-15T01:31:35 | 2026-11-15_hr_s7_hyde | index + search | 0.0430 | 1.2573 |
| 2026-11-15T02:10:48 | 2026-11-15_hr_s6_compressed | answers (20 queries) | 0.0203 | 1.2776 |
| 2026-11-15T02:38:48 | 2026-11-15_hr_s6_compressed | answers (318 queries) | 0.3954 | 1.6730 |
| 2026-11-15T02:45:43 | 2026-11-15_hr_pilot_agent | index + search | 0.0151 | 1.6882 |
| 2026-11-15T03:19:48 | 2026-11-15_hr_s8_agent | index + search | 0.2539 | 1.9421 |
| 2026-11-15T03:23:55 | 2026-11-15_hr_s8_agent | answers (318 queries) | 0.3170 | 2.2591 |
| 2026-11-15T03:26:32 | 2026-11-15_hr_trace_demo | index + search | 0.0020 | 2.2610 |
| 2026-11-15T03:29:36 | 2026-11-15_hr_robustness | abstention, injection, defended answers | 0.3982 | 2.6592 |
| 2026-11-15T03:42:29 | 2026-11-15_hr_robustness | abstention, injection, defended answers | 0.3087 | 2.9680 |
| 2026-11-15T04:07:52 | 2026-11-15_fin_baseline | index + search | 0.0524 | 3.0204 |
| 2026-11-15T04:10:58 | 2026-11-15_fin_baseline | answers (309 queries) | 0.3779 | 3.3983 |
| 2026-11-15T04:45:40 | 2026-11-15_fin_loo_no_headers | index + search | 0.2038 | 3.6021 |
| 2026-11-15T05:19:42 | 2026-11-15_fin_final | index + search | 1.4260 | 5.0281 |
| 2026-11-15T05:37:25 | 2026-11-15_fin_final_agent | index + search | 0.2267 | 5.2547 |
| 2026-11-15T05:44:25 | 2026-11-15_fin_final_agent | answers (309 queries) | 0.2982 | 5.5529 |
| 2026-11-15T06:01:24 | 2026-11-15_fin_loo_no_bm25 | index + search | 0.2268 | 5.7797 |
| 2026-11-15T06:13:56 | 2026-11-15_fin_loo_no_visual | index + search | 0.2323 | 6.0120 |
| 2026-11-15T06:14:56 | 2026-11-15_fin_loo_no_compression | index + search | 0.2325 | 6.2444 |
| 2026-11-15T10:47:57 | cache_check | prompt-cache check for contextual headers | 0.0047 | 6.2491 |
| 2026-11-15T10:49:25 | cache_check | prompt-cache check for contextual headers (2) | 0.0032 | 6.2523 |
| 2026-11-15T10:56:11 | 2026-11-15_hr_s9_agent_visual | index + search | 0.2410 | 6.4933 |
| 2026-11-15T11:00:27 | 2026-11-15_hr_s9_agent_visual | answers (318 queries) | 0.3124 | 6.8056 |

Total: 6.81 USD of a 25 USD cap.
