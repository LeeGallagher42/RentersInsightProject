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

## Verification on 2026-08-31
- JavaScript extracted from the HTML parses successfully with Node `new Function`.
- Static smoke checks verify core product strings and analytics event names.
- Local HTTP serving was attempted in the automation container; the server process runs, but the container's loopback fetch returned status 000, so network verification should be performed with deployment/browser tooling when available.

## Next build priorities before launch
1. Improve public-facing creator trust/safety copy and empty/error states.
2. Add lightweight collection/bookmark surface and a moderation transparency page/section.
3. Add structured metadata and richer social preview fields.
4. Verify interactive flows in a real browser against a Vercel preview.
5. At/after the launch deadline, deploy the best working version and verify the public URL.

## Persistence
This branch is intentionally isolated from the repository's main product work: `creator-platform-build`.
