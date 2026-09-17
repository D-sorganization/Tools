Awesome, I will replace all of these occurrences of `Math.min(...array)` and `indexOf` in `wedgeGroundClearance.ts`!
This will make a great PR!

Let's summarize the changes:
1. At line 152 in `firstContact`:
```ts
  let timeS: number | null = null;
  let candidateIndex = -1;
  // ⚡ Bolt Optimization: Use single-pass loop instead of Math.min(...array) and .indexOf()
  let initialMinimum = clearanceRows[0][0];
  let initialCandidateIndex = 0;
  for (let i = 1; i < clearanceRows[0].length; i++) {
    if (clearanceRows[0][i] < initialMinimum) {
      initialMinimum = clearanceRows[0][i];
      initialCandidateIndex = i;
    }
  }
  if (initialMinimum <= 1e-10) {
    timeS = times[0];
    candidateIndex = initialCandidateIndex;
```

2. At line 208-213 in `ballMetrics`:
```ts
  const state = candidateState(samples, candidates, ballTime);
  // ⚡ Bolt Optimization: Use single-pass loops instead of chained map/filter and array spreads
  let leadingMin = Infinity;
  let soleMin = Infinity;
  for (let i = 0; i < state.clearances.length; i++) {
    if (candidates[i].feature.startsWith("leading_edge")) {
      if (state.clearances[i] < leadingMin) leadingMin = state.clearances[i];
    } else {
      if (state.clearances[i] < soleMin) soleMin = state.clearances[i];
    }
  }

  let minimumPreBall = Infinity;
  for (let i = 0; i < envelope.length; i++) {
    if (envelope[i].timeS <= ballTime && envelope[i].minimumClearanceM < minimumPreBall) {
      minimumPreBall = envelope[i].minimumClearanceM;
    }
  }
  for (let i = 0; i < state.clearances.length; i++) {
    if (state.clearances[i] < minimumPreBall) {
      minimumPreBall = state.clearances[i];
    }
  }
```
And replace the references `Math.min(...leading)` and `Math.min(...sole)` with `leadingMin` and `soleMin`.

3. At line 253-260 in `wedgeGroundClearance`:
```ts
  const envelope = states.map((state, index) => {
    // ⚡ Bolt Optimization: Replace Math.min(...spread) and indexOf with a single-pass loop
    let minimum = state.clearances[0];
    let candidateIndex = 0;
    for (let i = 1; i < state.clearances.length; i++) {
      if (state.clearances[i] < minimum) {
        minimum = state.clearances[i];
        candidateIndex = i;
      }
    }
    return {
      timeS: times[index], minimumClearanceM: minimum,
      feature: candidates[candidateIndex].feature, worldPointM: state.points[candidateIndex],
    };
  });
```

This perfectly hits all the points:
- Memory: "always replace chained array methods like `.map(Math.abs)` combined with `Math.max(...spread)` and `.indexOf()` with standard single-pass `for` loops. This prevents allocating multiple intermediate arrays per frame and hitting the call stack, significantly reducing garbage collection pressure even for small array sizes."
- Bolt Guidelines: "Replace chained array methods + spread bounds with single-pass loops".

Let's generate the diff to apply and run tests in `/app/src/rate_of_closure/web/`
