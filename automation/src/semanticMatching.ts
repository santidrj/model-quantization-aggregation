import { config } from "./config.js";
import { normalizeLabel } from "./utils.js";

export type SemanticMap = Record<string, string>;

function buildNormalizedMap(raw: SemanticMap): Map<string, string> {
  const map = new Map<string, string>();
  for (const [alias, canonical] of Object.entries(raw)) {
    map.set(normalizeLabel(alias), canonical.trim());
  }
  return map;
}

/**
 * Configurable semantic equivalence: aliases resolve to a canonical (more generic) term.
 * Matching is case-insensitive; callers may pass an extended map.
 */
export class SemanticMatcher {
  private readonly map: Map<string, string>;

  constructor(raw: SemanticMap = config.semanticEquivalence) {
    this.map = buildNormalizedMap(raw);
  }

  /** Canonical label if this label is an alias; otherwise the original trimmed label. */
  canonicalOf(label: string): string {
    const hit = this.map.get(normalizeLabel(label));
    return hit ?? label.trim();
  }

  /** True when both labels denote the same concept via exact or mapped equality. */
  areEquivalent(a: string, b: string): boolean {
    const ca = normalizeLabel(this.canonicalOf(a));
    const cb = normalizeLabel(this.canonicalOf(b));
    if (ca === cb) return true;
    if (normalizeLabel(a) === normalizeLabel(b)) return true;
    // Plural/singular light touch: compare without trailing "s"
    const stripS = (s: string) => s.replace(/s\b/g, "").replace(/\s+/g, " ").trim();
    return stripS(ca) === stripS(cb);
  }

  /**
   * If `label` is an alias of a known canonical, return that canonical.
   * If `label` is already a canonical value in the map, return it.
   */
  mappedCanonical(label: string): string | null {
    const direct = this.map.get(normalizeLabel(label));
    if (direct) return direct;
    for (const canonical of this.map.values()) {
      if (normalizeLabel(canonical) === normalizeLabel(label)) return canonical;
    }
    return null;
  }

  isAlias(label: string): boolean {
    return this.map.has(normalizeLabel(label));
  }

  /** Labels that map to the same canonical as `label` (including itself if canonical). */
  equivalenceClass(label: string): string[] {
    const canon = normalizeLabel(this.canonicalOf(label));
    const out = new Set<string>();
    out.add(this.canonicalOf(label));
    for (const [aliasNorm, canonical] of this.map.entries()) {
      if (normalizeLabel(canonical) === canon) {
        out.add(canonical);
        // recover original alias casing from config keys when possible
        for (const [rawAlias, rawCanon] of Object.entries(config.semanticEquivalence)) {
          if (normalizeLabel(rawAlias) === aliasNorm) out.add(rawAlias);
          if (normalizeLabel(rawCanon) === canon) out.add(rawCanon);
        }
      }
    }
    return [...out];
  }
}

export const defaultSemanticMatcher = new SemanticMatcher(config.semanticEquivalence);
