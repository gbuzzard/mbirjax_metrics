I want to investigate fine-grained performance profiling of the mbirjax projectors using JAX's profiler and NVIDIA's GPU profiling tools. The goal is to understand where time and memory actually go inside the forward / back / VCD projection kernels — a level deeper than the coarse min-time and peak-memory the regression harness records today — so we can see what's worth instrumenting or optimizing.

Start by reading .claude/dashboard_orientation.md for the lay of the land: the two-repo setup, the measurement engine, and what the projectors are and how they're currently measured. The library under test is the sibling mbirjax repo — the projector code lives there.

Please also read
 * /Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/.claude/claude_prompt.md
 * /Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/.claude/lessons.md
 * /Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/.claude/back_projection_overview.md

This is exploratory; don't change the harness or the library yet. After reading the orientation doc and skimming the projector code, come back with:
 * a short summary of how a projector run is set up today and where profiling could attach,
 * the JAX and NVIDIA profiling approaches worth considering, with their trade-offs across the CPU (Mac) and GPU (Gautschi H100) targets, 
 * and a proposed first experiment — the smallest thing that would produce a useful fine-grained trace of one projector.

Please ask before installing profiling tooling or running anything heavy on the cluster.