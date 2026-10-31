const state = { result: null };
const $ = (id) => document.getElementById(id);
const palette = { negative: '#9b4938', neutral: '#8a7966', positive: '#526b4b' };
const MODEL_NAME = 'TF-IDF + Logistic Regression (browser)';

function cleanText(text) {
  return String(text).toLowerCase()
    .replace(/http\S+|www\.\S+/g, ' ')
    .replace(/@\w+/g, ' ').replace(/#/g, ' ')
    .replace(/[^a-z0-9\s'!?.,]/g, ' ')
    .replace(/\s+/g, ' ').trim();
}

function features(text) {
  const tokens = cleanText(text).match(/[a-z0-9_]{2,}/g) || [];
  const counts = new Map();
  for (let i = 0; i < tokens.length; i += 1) {
    counts.set(tokens[i], (counts.get(tokens[i]) || 0) + 1);
    if (i > 0) {
      const bigram = `${tokens[i - 1]} ${tokens[i]}`;
      counts.set(bigram, (counts.get(bigram) || 0) + 1);
    }
  }
  const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
  return [...counts].map(([term, count]) => [JANMAT_MODEL.vocabulary[term], Math.log1p(count) / total])
    .filter(([index]) => index !== undefined);
}

function predictComment(comment) {
  const scores = JANMAT_MODEL.intercepts.slice();
  features(comment).forEach(([index, value]) => {
    for (let label = 0; label < scores.length; label += 1) {
      scores[label] += value * JANMAT_MODEL.idf[index] * JANMAT_MODEL.coefficients[label][index];
    }
  });
  const max = Math.max(...scores);
  const probabilities = scores.map((score) => Math.exp(score - max));
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  const confidence = Math.max(...probabilities) / total;
  const index = probabilities.indexOf(Math.max(...probabilities));
  return { sentiment: JANMAT_MODEL.labels[index], confidence: Number(confidence.toFixed(3)) };
}

function parseCsv(text) {
  const records = []; let row = []; let cell = ''; let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"' && quoted && text[i + 1] === '"') { cell += '"'; i += 1; }
    else if (char === '"') quoted = !quoted;
    else if (char === ',' && !quoted) { row.push(cell); cell = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(cell); if (row.some(Boolean)) records.push(row); row = []; cell = '';
    } else cell += char;
  }
  if (cell || row.length) { row.push(cell); records.push(row); }
  const headers = records.shift() || [];
  return records.map((values) => Object.fromEntries(headers.map((header, i) => [header, values[i] || ''])));
}

function rowsFromSource() {
  if ($('source').value === 'sample') return JANMAT_SAMPLE.map((row) => ({ comment: row.comment, sourceTopic: row.true_topic || '' }));
  const file = $('file').files[0];
  if (!file) throw new Error('Choose a CSV record first.');
  return file.text().then((text) => {
    const records = parseCsv(text);
    const column = Object.keys(records[0] || {}).find((key) => /comment|text|content|feedback/i.test(key))
      || Object.keys(records[0] || {})[0];
    if (!column) throw new Error('The CSV must contain a text column.');
    return records.map((row) => ({ comment: row[column], sourceTopic: row.true_topic || row.topic || '' }))
      .filter((row) => row.comment.trim());
  });
}

function setStatus(text) { $('status').textContent = text; }
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }

function render(result) {
  state.result = result; $('empty').hidden = true; $('results').hidden = false;
  const total = result.total;
  $('mood').innerHTML = ['negative', 'neutral', 'positive'].map((label) => {
    const count = result.counts[label]; const width = count / total * 100;
    return `<div class="mood-segment" style="width:${width}%;background:${palette[label]}">${Math.round(width)}% ${label}<small>${count}</small></div>`;
  }).join('');
  $('facts').innerHTML = [[total.toLocaleString(), 'comments analysed'], [result.topics.length, 'themes found'],
    [`${Math.round(result.average_confidence * 100)}%`, 'average confidence'],
    [result.rows.filter((row) => row.confidence < .6).length, 'comments to review']]
    .map(([value, label]) => `<div class="fact"><strong>${value}</strong><small>${label}</small></div>`).join('');
  $('concerns').textContent = result.summary.concerns || 'Not enough comments to summarise.';
  $('support').textContent = result.summary.support || 'Not enough comments to summarise.';
  $('topics-chart').innerHTML = result.topics.map((topic) => {
    const bar = ['negative', 'neutral', 'positive'].map((label) =>
      `<i class="${label}" style="width:${topic[label] / topic.total * 100}%"></i>`).join('');
    return `<div class="topic-row"><span class="topic-name" title="${escapeHtml(topic.name)}">${escapeHtml(topic.name)}</span><div class="bar">${bar}</div><span>${topic.total}</span></div>`;
  }).join('');
  renderRows(); setStatus(`${total} comments · ${result.model}`);
}

function renderRows() {
  if (!state.result) return;
  const sentiment = $('sentiment-filter').value;
  const search = $('search').value.toLowerCase();
  const rows = state.result.rows.filter((row) =>
    (!sentiment || row.sentiment === sentiment) && (!search || row.comment.toLowerCase().includes(search)));
  $('rows').innerHTML = rows.map((row) => `<tr><td>${escapeHtml(row.comment)}</td><td><span class="badge ${row.sentiment}">${row.sentiment}</span></td><td>${Math.round(row.confidence * 100)}%</td><td>${escapeHtml(row.topic)}</td></tr>`).join('');
}

function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[char])); }

function makeTopics(rows) {
  const groups = new Map();
  rows.forEach((row) => { const name = row.sourceTopic || 'General consultation'; if (!groups.has(name)) groups.set(name, []); groups.get(name).push(row); });
  return [...groups].map(([name, subset]) => ({ name, total: subset.length, ...Object.fromEntries(['negative', 'neutral', 'positive'].map((label) => [label, subset.filter((row) => row.sentiment === label).length])) }))
    .sort((a, b) => b.total - a.total);
}

async function analyze() {
  showError(''); setStatus('Reading the record…'); $('analyze').disabled = true;
  try {
    const comments = await rowsFromSource();
    if (comments.length < 1) throw new Error('The CSV does not contain any comments.');
    const rows = comments.map((item) => ({ ...item, ...predictComment(item.comment) }));
    const counts = Object.fromEntries(['negative', 'neutral', 'positive'].map((label) => [label, rows.filter((row) => row.sentiment === label).length]));
    const negative = rows.filter((row) => row.sentiment === 'negative').slice(0, 3).map((row) => row.comment).join(' ');
    const positive = rows.filter((row) => row.sentiment === 'positive').slice(0, 3).map((row) => row.comment).join(' ');
    render({ model: MODEL_NAME, total: rows.length, average_confidence: rows.reduce((sum, row) => sum + row.confidence, 0) / rows.length,
      counts, topics: makeTopics(rows), summary: { concerns: negative, support: positive }, rows });
  } catch (error) { showError(error.message); setStatus('Could not read the record'); }
  finally { $('analyze').disabled = false; }
}

function classify() {
  const comment = $('single-comment').value.trim(); if (!comment) return;
  const data = predictComment(comment); const result = $('single-result');
  result.hidden = false; result.className = `single-result ${data.sentiment}`;
  result.innerHTML = `<b>${data.sentiment[0].toUpperCase() + data.sentiment.slice(1)}</b> · ${Math.round(data.confidence * 100)}% confidence`;
}

function download() {
  if (!state.result) return;
  const header = ['comment', 'sentiment', 'confidence', 'topic'];
  const csv = [header, ...state.result.rows.map((row) => header.map((key) => `"${String(row[key] || '').replaceAll('"', '""')}"`))]
    .map((row) => row.join(',')).join('\n');
  const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  link.download = 'janmat_results.csv'; link.click(); URL.revokeObjectURL(link.href);
}

$('model').add(new Option(MODEL_NAME, MODEL_NAME));
$('source').addEventListener('change', () => { const upload = $('source').value === 'upload'; $('file').hidden = !upload; $('file-name').textContent = upload ? 'No file selected' : 'Using the built-in consultation record'; });
$('file').addEventListener('change', () => { $('file-name').textContent = $('file').files[0]?.name || 'No file selected'; });
$('analyze').addEventListener('click', analyze); $('classify').addEventListener('click', classify); $('download').addEventListener('click', download);
$('sentiment-filter').addEventListener('change', renderRows); $('search').addEventListener('input', renderRows);
