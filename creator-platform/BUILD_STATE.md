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

## Public product surfaces now present
- `trust.html`: Trust & Safety experience covering original-work requirements, mature-content labeling, reports, moderation outcomes and creator appeals.
- `creator-guide.html`: creator onboarding with live 90/10 calculator and publish checklist.
- `library.html`: browser-persistent personal library that reads the same liked-work, followed-creator and analytics state as the main app; includes saved work search, remove/clear actions, following view and recent local activity.
- Clean Vercel routes for `/trust`, `/creator-guide` and `/library`.
- `manifest.webmanifest` and `robots.txt` foundations for a public installable/indexable web product.

## Verification / persistence
- Main-app JavaScript has been parsed successfully with Node `new Function` in prior runs and core product strings/events smoke-tested.
- Supporting pages are dependency-free HTML/CSS/JS and share INKSIDE browser storage keys where relevant.
- Deployment configuration is valid JSON and includes baseline security headers plus clean public routes.
- All changes are preserved on isolated branch `creator-platform-build`; the repository's main product branch is untouched.

## Latest product increment — 2026-09-01
- Added a real Library surface instead of another planning-only pass.
- Library exposes persistent saved works, followed creators and recent analytics activity using the main app's existing `inkside_likes`, `inkside_follows`, `inkside_published`, and `inkside_events` state.
- Added search, per-work removal, clear-saved control and responsive card layouts.
- Added analytics-ready `page_view`, `library_view`, `library_remove`, and `library_clear` events through the existing `inkside:analytics` pattern.
- Added `/library` rewrite to Vercel configuration.

## Next build priorities before launch
1. Link Trust & Safety, Creator Guide and Library directly from the main app navigation/footer.
2. Add richer discovery sorting and explicit saved/bookmark control in addition to likes.
3. Add richer OpenGraph/social metadata to the main app and public surfaces.
4. Verify main app + `/trust` + `/creator-guide` + `/library` interactively in a deployed browser preview.
5. At/after the launch deadline, deploy the best working version and verify the public URL.

## Persistence
This branch is intentionally isolated from the repository's main product work: `creator-platform-build`.
