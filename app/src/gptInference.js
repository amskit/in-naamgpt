/**
 * Browser-side GPT inference engine.
 * Mirrors the Python microGPT algorithm exactly.
 */

export function softmax(values) {
  if (!values.length) return []
  const max = Math.max(...values)
  const exps = values.map((v) => Math.exp(v - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  if (!Number.isFinite(sum) || sum <= 0) return values.map(() => 1 / values.length)
  return exps.map((v) => v / sum)
}

export function dotProduct(a, b) {
  let total = 0
  for (let i = 0; i < a.length; i++) total += a[i] * b[i]
  return total
}

export function matVec(matrix, vector) {
  return matrix.map((row) => dotProduct(row, vector))
}

export function rmsNorm(x, eps = 1e-5) {
  const ms = x.reduce((a, v) => a + v * v, 0) / x.length
  const scale = 1 / Math.sqrt(ms + eps)
  return x.map((v) => v * scale)
}

export function relu(x) {
  return x.map((v) => Math.max(0, v))
}

/**
 * Run one token through the GPT model.
 * Returns logits for the next token.
 */
export function gptForward(tokenId, posId, keysCache, valuesCache, snapshot) {
  const { wte, wpe, attention, mlp, lm_head, n_embd } = snapshot
  const { n_head, head_dim, attn_wq, attn_wk, attn_wv, attn_wo } = attention
  const { mlp_fc1, mlp_fc2 } = mlp

  // Embedding
  let x = wte[tokenId].map((t, i) => t + wpe[posId][i])
  x = rmsNorm(x)

  // Attention block
  const xResidual1 = x
  x = rmsNorm(x)

  const q = matVec(attn_wq, x)
  const k = matVec(attn_wk, x)
  const v = matVec(attn_wv, x)

  keysCache.push(k)
  valuesCache.push(v)

  const xAttn = []
  for (let h = 0; h < n_head; h++) {
    const hs = h * head_dim
    const qH = q.slice(hs, hs + head_dim)

    const attnLogits = []
    for (let t = 0; t < keysCache.length; t++) {
      const kH = keysCache[t].slice(hs, hs + head_dim)
      let score = 0
      for (let j = 0; j < head_dim; j++) score += qH[j] * kH[j]
      attnLogits.push(score / Math.sqrt(head_dim))
    }

    const attnWeights = softmax(attnLogits)

    for (let j = 0; j < head_dim; j++) {
      let val = 0
      for (let t = 0; t < valuesCache.length; t++) {
        val += attnWeights[t] * valuesCache[t][hs + j]
      }
      xAttn.push(val)
    }
  }

  x = matVec(attn_wo, xAttn)
  x = x.map((a, i) => a + xResidual1[i])

  // MLP block
  const xResidual2 = x
  x = rmsNorm(x)
  x = relu(matVec(mlp_fc1, x))
  x = matVec(mlp_fc2, x)
  x = x.map((a, i) => a + xResidual2[i])

  return matVec(lm_head, x)
}

/**
 * Generate a single name from the model.
 */
export function generateName(snapshot, temperature = 0.5) {
  const { tokenizer, block_size } = snapshot
  const { uchars, bos } = tokenizer
  const vocabSize = uchars.length + 1

  const keysCache = []
  const valuesCache = []
  let tokenId = bos
  const chars = []

  for (let pos = 0; pos < block_size; pos++) {
    const logits = gptForward(tokenId, pos, keysCache, valuesCache, snapshot)

    // Apply temperature
    const scaledLogits = logits.map((l) => l / temperature)
    const probs = softmax(scaledLogits)

    // Sample from distribution
    const r = Math.random()
    let cumulative = 0
    let nextToken = 0
    for (let i = 0; i < vocabSize; i++) {
      cumulative += probs[i]
      if (r < cumulative) {
        nextToken = i
        break
      }
    }

    if (nextToken === bos) break
    chars.push(uchars[nextToken])
    tokenId = nextToken
  }

  // NFC normalize the NFD characters back to readable Devanagari
  const nfdText = chars.join('')
  const name = nfdText.normalize('NFC')
  return { name, nfdText, chars, probs: [] }
}

/**
 * Generate a name with step-by-step data for visualization.
 */
export function generateNameWithSteps(snapshot, temperature = 0.5) {
  const { tokenizer, block_size } = snapshot
  const { uchars, bos } = tokenizer
  const vocabSize = uchars.length + 1

  const keysCache = []
  const valuesCache = []
  let tokenId = bos
  const chars = []
  const steps = []

  for (let pos = 0; pos < block_size; pos++) {
    const logits = gptForward(tokenId, pos, keysCache, valuesCache, snapshot)
    const scaledLogits = logits.map((l) => l / temperature)
    const probs = softmax(scaledLogits)

    // Get top 5 predictions
    const topK = probs
      .map((p, i) => ({ prob: p, tokenId: i, char: i === bos ? '[END]' : uchars[i] }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5)

    const r = Math.random()
    let cumulative = 0
    let nextToken = 0
    for (let i = 0; i < vocabSize; i++) {
      cumulative += probs[i]
      if (r < cumulative) {
        nextToken = i
        break
      }
    }

    steps.push({
      position: pos,
      inputToken: tokenId,
      inputChar: tokenId === bos ? '[BOS]' : uchars[tokenId],
      topK,
      selectedToken: nextToken,
      selectedChar: nextToken === bos ? '[END]' : uchars[nextToken],
    })

    if (nextToken === bos) break
    chars.push(uchars[nextToken])
    tokenId = nextToken
  }

  const nfdText = chars.join('')
  const name = nfdText.normalize('NFC')
  return { name, nfdText, chars, steps }
}
