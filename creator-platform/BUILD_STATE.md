# INKSIDE Creator Platform — build state

## Current working product
- Single-file static web app at `creator-platform/index.html`.
- Responsive discovery grid for Art, Comics and Free works.
- Search across work title, description and creator identity.
- Six original seeded demo works rendered as inline SVG artwork.
- Work detail experience with comic page metadata, free/paid intent, creator economics and sharing.
- Creator profile views with catalog, followers and follow state.
- Creator Studio dashboard with audience and earnings snapshot.
- Publish flow for art/comics with price, page count, optional local image preview, rights confirmation and guidelines confirmation.
- Browser-persistent likes, follows, published demo works, moderation reports and analytics events.
- Moderation/report flow queues reports locally without automatic takedown.
- Analytics-ready event dispatch via `inkside:analytics`, plus bounded local event history.
- Explicit creator economics: 90% creator / 10% platform, €0 listing fee in MVP messaging, and interactive revenue calculator.
- Keyboard card activation, Escape-to-close modals and responsive mobile breakpoints.
- `vercel.json` adds baseline security headers.

## Added in the latest product run
- New `trust.html` public-facing Trust & Safety experience explaining original-work requirements, mature-content labeling, report review, moderation outcomes and creator appeals.
- New `creator-guide.html` onboarding surface with a live 90/10 earnings calculator, draft-to-publish guidance and a creator publishing checklist.
- Trust and creator-guide pages emit analytics-ready view events using the same `inkside:analytics` event pattern.
- Added clean Vercel routes for `/trust` and `/creator-guide`.
- Added `manifest.webmanifest` and `robots.txt` foundations for a public installable/indexable web product.
- Strengthened deployment headers with `Cross-Origin-Opener-Policy: same-origin` while preserving existing security headers.

## Verification on 2026-08-31
- Existing main app JavaScript was previously parsed successfully with Node `new Function` and core product strings/events smoke-tested.
- New trust and creator-guide pages are static, dependency-free HTML/CSS/JS with no external assets.
- Vercel configuration remains valid JSON and now includes clean routes for both new public-facing pages.
- All changes are preserved on isolated branch `creator-platform-build`; the repository's main product branch is untouched.

## Next build priorities before launch
1. Link Trust & Safety and Creator Guide directly from the main app navigation/footer.
2. Add lightweight collection/bookmark surface and stronger discovery sorting.
3. Add richer OpenGraph/social metadata to the main app and creator/work views.
4. Verify main app + `/trust` + `/creator-guide` interactively in a deployed browser preview.
5. At/after the launch deadline, deploy the best working version and verify the public URL.

## Persistence
This branch is intentionally isolated from the repository's main product work: `creator-platform-build`.
