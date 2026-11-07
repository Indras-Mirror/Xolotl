#!/usr/bin/env bash
# merge-xolotl.sh - drive the unification of the Strata engine family into one tree.
# See MERGE_SCOPE.md for the measured conflict surface (tools/ds4 option, expected hunks, gates).
#
#   tools/xolotl/merge-xolotl.sh            # dry run: report what would conflict (default)
#   tools/xolotl/merge-xolotl.sh --apply    # actually merge, leaving conflicts for review
#
# Refuses to run on a dirty tree. Never pushes. Aborting a merge is `git merge --abort`.
set -u
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1

cd "$(git rev-parse --show-toplevel)" || exit 2
BR=$(git branch --show-current)
[ "$BR" = "xolotl" ] || { echo "refusing: on branch '$BR', expected 'xolotl'" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "refusing: working tree is dirty" >&2; exit 2; }

# the sources: origin/glm and origin/mimo in this clone (origin = ~/AI/Xolotl)
for r in origin/glm origin/mimo; do
    git rev-parse --verify -q "$r" >/dev/null || { echo "missing ref $r - fetch first" >&2; exit 2; }
done

echo "== base =="; git log -1 --oneline | cat
for r in origin/glm origin/mimo; do
    echo "== $r =="
    n=$(git merge-tree --write-tree --name-only "$BR" "$r" 2>/dev/null | grep -c '^CONFLICT') || true
    echo "   conflicts: ${n:-0}"
    git merge-tree --write-tree --name-only "$BR" "$r" 2>/dev/null | grep '^CONFLICT' | sed 's/CONFLICT (content): Merge conflict in /   /' | head -20
done

[ "$APPLY" = 1 ] || { echo; echo "dry run only - re-run with --apply to merge"; exit 0; }

# NOTE: merge order matters.  Drop legacy tools/ds4/ds4_* BEFORE merging mimo if Mal approves it
# (docs/history/MERGE_SCOPE-2026-10-16.md SEC 6 - it removes ~30 of mimo's 46 hunks).  Kernels first, CMakeLists last.
for r in origin/glm origin/mimo; do
    echo "== merging $r =="
    if git merge --no-commit --no-ff "$r"; then
        echo "   CLEAN - review, then: git commit"
    else
        echo "   CONFLICTS - resolve, then 'git add' + 'git commit'; 'git merge --abort' to bail"
        git diff --name-only --diff-filter=U | sed 's/^/     U /'
    fi
    break   # one at a time: resolve + gate before the next
done
