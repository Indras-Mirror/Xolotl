/**
 * Search through a service the person has their own key for: Exa or Perplexity (the Harness ships both as separate packages the app's
 * bundle leaves out, so these are the same two small calls written here; DeepSeek's provider is in the bundle and is used as it is).
 * Shapes follow the Harness's providers (packages/web/web-search-exa and web-search-perplexity at the pinned commit). Keirolabs has no
 * Harness package; its call is Keirolabs' lite search (titles, links and snippets, billed per request whatever the count).
 * Keenable has no Harness package either; unlike the others it also searches without a key (its public endpoint).
 */
import { SearchUnavailable } from './web-search-ddg.mjs';

const abortError = e => e?.name === 'AbortError';

async function call(service, url, options, signal) {
  let res;
  try { res = await fetch(url, { ...options, redirect: 'error', ...(signal ? { signal } : {}) }); }
  catch (error) {
    if (abortError(error)) throw new SearchUnavailable('The search was cancelled.', 'WEB_ABORTED', { cause: error });
    throw new SearchUnavailable(`The search could not reach ${service}. Check the internet connection and try again.`, 'WEB_PROVIDER_ERROR', { cause: error });
  }
  if (!res.ok) {
    let detail = '';
    try { const body = await res.json(); detail = typeof body.error === 'string' ? body.error : body.error?.message ?? body.message ?? ''; } catch { /* not JSON */ }
    if (res.status === 401 || res.status === 403) throw new SearchUnavailable(`${service} did not accept the key. Check it in Settings, Tools.`, 'WEB_PROVIDER_ERROR');
    if (res.status === 429) throw new SearchUnavailable(`${service} says there have been too many searches. Try again in a minute.`, 'WEB_RATE_LIMITED');
    throw new SearchUnavailable(`${service} could not search (error ${res.status}).${detail ? ' ' + String(detail).slice(0, 200) : ''}`, 'WEB_PROVIDER_ERROR');
  }
  try { return await res.json(); }
  catch (error) { throw new SearchUnavailable(`${service} sent an answer that could not be read.`, 'WEB_PROVIDER_ERROR', { cause: error }); }
}

export function createExaSearch({ key }) {
  return {
    id: 'exa',
    available: () => typeof key === 'function' ? !!key() : !!key,
    async search(request, signal) {
      const apiKey = typeof key === 'function' ? key() : key;
      const body = await call('Exa', 'https://api.exa.ai/search', {
        method: 'POST',
        headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({ query: request.query, type: 'auto', contents: { highlights: { highlightsPerUrl: 1 } }, ...(request.maxResults ? { numResults: request.maxResults } : {}) }),
      }, signal);
      const sources = (body.results ?? []).flatMap(r => {
        const snippet = (r.highlights ?? []).find(h => String(h).trim());
        return snippet === undefined ? [] : [{ url: r.url, ...(r.title ? { title: r.title } : {}), snippet, ...(r.publishedDate ? { publishedAt: r.publishedDate } : {}) }];
      });
      return { sources, truncated: false };
    },
  };
}

export function createPerplexitySearch({ key }) {
  return {
    id: 'perplexity',
    available: () => typeof key === 'function' ? !!key() : !!key,
    async search(request, signal) {
      const apiKey = typeof key === 'function' ? key() : key;
      const body = await call('Perplexity', 'https://api.perplexity.ai/chat/completions', {
        method: 'POST',
        headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({ model: 'sonar', max_tokens: 1024, messages: [{ role: 'user', content: request.query }] }),
      }, signal);
      const content = body.choices?.[0]?.message?.content;
      const sources = body.search_results !== undefined
        ? body.search_results.map(r => ({ url: r.url, ...(r.title ? { title: r.title } : {}), ...(r.snippet ? { snippet: r.snippet } : {}), ...(r.date ? { publishedAt: r.date } : {}) }))
        : (body.citations ?? []).map(url => ({ url }));
      return { ...(content ? { content } : {}), sources, truncated: false };
    },
  };
}

/**
 * Keenable is the one keyed service that also works without a key: the public endpoint takes the same body and answer and
 * asks only for the app's name in X-Keenable-Title (rate limited per address). A saved key switches to the keyed endpoint
 * (X-API-Key), which lifts those limits. Search only (titles, links, snippets); page fetch stays with the fetch provider.
 */
export function createKeenableSearch({ key }) {
  return {
    id: 'keenable',
    available: () => true,
    async search(request, signal) {
      const apiKey = typeof key === 'function' ? key() : key;
      const body = await call('Keenable', `https://api.keenable.ai/v1/search${apiKey ? '' : '/public'}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json', ...(apiKey ? { 'x-api-key': apiKey } : { 'x-keenable-title': 'Xolotl' }) },
        body: JSON.stringify({ query: request.query, ...(request.maxResults ? { max_results: request.maxResults } : {}) }),
      }, signal);
      const sources = (body.results ?? []).flatMap(r => r?.url
        ? [{ url: r.url, ...(r.title ? { title: r.title } : {}), ...(r.snippet ? { snippet: r.snippet } : {}), ...(r.published_at ? { publishedAt: r.published_at } : {}) }]
        : []);
      return { sources, truncated: false };
    },
  };
}

export function createKeirolabsSearch({ key }) {
  return {
    id: 'keirolabs',
    available: () => typeof key === 'function' ? !!key() : !!key,
    async search(request, signal) {
      const apiKey = typeof key === 'function' ? key() : key;
      const body = await call('Keirolabs', 'https://api.keirolabs.cloud/api/v2/search/lite', {
        method: 'POST',
        headers: { authorization: `Bearer ${apiKey}`, 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify({ query: request.query, ...(request.maxResults ? { maxResults: request.maxResults } : {}) }),
      }, signal);
      const sources = (body.results ?? []).flatMap(r => r?.url
        ? [{ url: r.url, ...(r.title ? { title: r.title } : {}), ...(r.snippet ? { snippet: r.snippet } : {}), ...(r.publishedDate ? { publishedAt: r.publishedDate } : {}) }]
        : []);
      return { sources, truncated: false };
    },
  };
}
