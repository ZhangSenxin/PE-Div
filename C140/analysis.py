"""Positive calling and exact 1-aa sequence families for the C140 Methods."""
from collections import Counter
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'


def result_columns(path, sheet, sequence_column, score_column):
    """Read just two columns, following XLSX relationships; bounded XML memory.

    Both inline strings and shared strings are supported. Empty labels,
    nonnumeric scores, duplicate labels and missing cells are errors.
    """
    with ZipFile(path) as book:
        sheets = ET.fromstring(book.read('xl/workbook.xml')).find(NS + 'sheets')
        target = next((s for s in sheets if s.attrib['name'] == sheet), None)
        if target is None:
            raise ValueError(f'{path}: missing sheet {sheet}')
        relation_id = target.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        relations = ET.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        target_path = next(r.attrib['Target'] for r in relations if r.attrib['Id'] == relation_id)
        import posixpath
        target_path = posixpath.normpath(target_path.lstrip('/') if target_path.startswith('/') else 'xl/' + target_path)
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            with book.open('xl/sharedStrings.xml') as stream:
                for event, element in ET.iterparse(stream, events=('end',)):
                    if element.tag == NS + 'si':
                        strings.append(''.join(t.text or '' for t in element.iter(NS + 't')))
                        element.clear()
        selected_columns = None
        sequences, values = [], []
        with book.open(target_path) as stream:
            iterator = ET.iterparse(stream, events=('start', 'end'))
            sheet_data = None
            for event, row in iterator:
                if event == 'start' and row.tag == NS + 'sheetData':
                    sheet_data = row
                if event != 'end' or row.tag != NS + 'row':
                    continue
                cells = {}
                for cell in row:
                    column = ''.join(c for c in cell.attrib.get('r', '') if c.isalpha())
                    if selected_columns is not None and column not in selected_columns:
                        continue
                    kind = cell.attrib.get('t')
                    value = cell.findtext(NS + 'v')
                    if kind == 's' and value is not None:
                        value = strings[int(value)]
                    elif kind == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.iter(NS + 't'))
                    cells[column] = value
                if selected_columns is None:
                    try:
                        selected_columns = [next(c for c, v in cells.items() if v == header)
                                            for header in (sequence_column, score_column)]
                    except StopIteration as error:
                        raise ValueError(f'{path}: required columns not found') from error
                elif cells:
                    sequence, value = (cells.get(c) for c in selected_columns)
                    if not sequence or value is None:
                        raise ValueError(f'{path}: missing sequence or score at row {row.attrib.get("r")}')
                    sequences.append(sequence)
                    values.append(float(value))
                if sheet_data is not None:
                    sheet_data.remove(row)
                row.clear()
    scores = np.asarray(values, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or len(set(sequences)) != len(sequences):
        raise ValueError(f'{path}: empty data, nonfinite scores or duplicate sequence labels')
    return sequences, scores





def hamming_matrix(sequences, expected_length):
    if not sequences or any(len(s) != expected_length for s in sequences):
        raise ValueError(f'Expected nonempty sequences of length {expected_length}')
    if not set(''.join(sequences)) <= set(AMINO_ACIDS):
        raise ValueError('Sequences must contain only the 20 standard amino acids')
    if expected_length > 255:
        raise ValueError('uint8 Hamming implementation requires length <=255')
    array = np.frombuffer(''.join(sequences).encode('ascii'), dtype=np.uint8).reshape(len(sequences), -1)
    distance = np.empty((len(array), len(array)), dtype=np.uint8)
    for start in range(0, len(array), 128):
        distance[start:start + 128] = np.count_nonzero(array[start:start + 128, None, :] != array[None, :, :], axis=2)
    return distance


def components(distance):
    """Transitive connected components of Hamming <=1; stable original IDs."""
    visited = np.zeros(len(distance), dtype=bool)
    clusters = []
    for start in range(len(distance)):
        if visited[start]:
            continue
        visited[start] = True
        queue, members = [start], []
        while queue:
            node = queue.pop()
            members.append(node)
            neighbors = np.flatnonzero((distance[node] <= 1) & ~visited)
            visited[neighbors] = True
            queue.extend(neighbors.tolist())
        clusters.append(sorted(members))
    return sorted(clusters, key=lambda c: (-len(c), c[0]))


def minimum_spanning_tree(distance):
    """Prim MST with original lexicographic tie breaks on the full union."""
    n = len(distance)
    used = np.zeros(n, bool)
    best = np.full(n, 256, dtype=np.int16)
    parent = np.full(n, -1)
    best[0] = 0
    edges = []
    for _ in range(n):
        node = int(np.argmin(np.where(used, 999, best)))
        used[node] = True
        if parent[node] >= 0:
            edges.append((int(parent[node]), node, int(best[node])))
        update = ~used & (distance[node] < best)
        best[update] = distance[node, update]
        parent[update] = node
    return edges


def family_data(selected, settings):
    sequences = sorted(set(selected['exp1']) | set(selected['exp2']))
    distance = hamming_matrix(sequences, settings['sequence_length'])
    union = []
    for index, sequence in enumerate(sequences):
        a, b = sequence in selected['exp1'], sequence in selected['exp2']
        union.append(dict(sequence_id=f'S{index + 1:05d}', seq=sequence,
                          category='overlap' if a and b else 'exp1_only' if a else 'exp2_only',
                          exp1_Z_total=selected['exp1'].get(sequence), exp2_Z_total=selected['exp2'].get(sequence)))
    families, memberships = [], []
    for index, members in enumerate(components(distance)):
        cluster_id = f'AA1_C{index + 1:04d}'
        count = Counter(union[i]['category'] for i in members)
        a, b, overlap = (count[c] for c in ('exp1_only', 'exp2_only', 'overlap'))
        n = len(members)
        asop = max(a, b) / n
        signed = (1 if a > b else -1 if b > a else 0) * asop
        families.append(dict(cluster_id=cluster_id, total_nodes=n, exp1_only=a, exp2_only=b,
                             overlap=overlap, exclusive_nodes=a + b, ASOP=asop, signed_ASOP=signed,
                             included=n >= settings['minimum_plot_nodes'],
                             plotted=n >= settings['minimum_plot_nodes'] and a + b > 0, indices=members))
        memberships.extend(dict(cluster_id=cluster_id, **union[i]) for i in members)
    edges = minimum_spanning_tree(distance)
    # Cutting the full MST at 1 aa must yield the same connected components.
    mapping = {i: f['cluster_id'] for f in families for i in f['indices']}
    retained = [(u, v, w) for u, v, w in edges if w <= 1]
    if len(retained) != len(union) - len(families) or any(mapping[u] != mapping[v] for u, v, _ in retained):
        raise AssertionError('MST and connected-component families disagree')
    return dict(union=union, families=families, memberships=memberships, global_mst_edges=edges)


def select_examples(families, targets):
    plotted = [f for f in families if f['plotted']]
    if not plotted:
        return []
    return [dict(target_x=target, **min(plotted, key=lambda f: (
        abs(f['signed_ASOP'] - target), -f['exclusive_nodes'], -f['total_nodes'], f['cluster_id']))) for target in targets]
