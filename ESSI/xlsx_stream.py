"""使用 Artifact Tool 生成的样式，流式写入标准 XLSX；无需第三方依赖。"""
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'


def column(n):
    name=''
    while n:
        n,r=divmod(n-1,26)
        name=chr(65+r)+name
    return name


def cell(ref,value,header=False):
    if value is None:
        return f'<c r="{ref}" s="3"/>'
    if isinstance(value,bool):
        return f'<c r="{ref}" s="3" t="b"><v>{int(value)}</v></c>'
    if isinstance(value,(int,float)):
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError(f'{ref}: 不允许非有限数值')
        return f'<c r="{ref}" s="{8 if isinstance(value,int) else 4}"><v>{value!r}</v></c>'
    value=str(value)
    if len(value)>32767 or re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',value):
        raise ValueError(f'{ref}: 文本超长或含无效控制字符')
    return f'<c r="{ref}" s="{2 if header else 3}" t="inlineStr"><is><t xml:space="preserve">{escape(value)}</t></is></c>'


def write_workbook(path,sheets):
    """sheets={页名: (dict行列表, 有序列名列表)}；返回可回读核对的摘要。"""
    path=Path(path)
    if path.exists():
        raise ValueError(f'输出已存在，拒绝覆盖: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    names=list(sheets)
    if not names or len(names)!=len(set(names)):
        raise ValueError('工作簿至少需要一个唯一工作表')
    for name in names:
        if not name or len(name)>31 or re.search(r'[\\/?*\[\]:]',name):
            raise ValueError('非法Excel工作表名: '+name)
    receipt=[]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        overrides=''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1,len(names)+1))
        z.writestr('[Content_Types].xml','<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'+overrides+'</Types>')
        z.writestr('_rels/.rels',f'<Relationships xmlns="{PKG}"><Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        sheet_xml=''.join(f'<sheet name={quoteattr(name)} sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(names,1))
        z.writestr('xl/workbook.xml',f'<workbook xmlns="{NS}" xmlns:r="{REL}"><bookViews><workbookView/></bookViews><sheets>{sheet_xml}</sheets></workbook>')
        relations=''.join(f'<Relationship Id="rId{i}" Type="{REL}/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(names)+1))
        z.writestr('xl/_rels/workbook.xml.rels',f'<Relationships xmlns="{PKG}">{relations}<Relationship Id="rId{len(names)+1}" Type="{REL}/styles" Target="styles.xml"/></Relationships>')
        z.writestr('xl/styles.xml',(Path(__file__).parent/'assets/xlsx_styles.xml').read_bytes())
        for i,(name,(rows,fields)) in enumerate(sheets.items(),1):
            if len(rows)>1048575 or len(fields)>16384:
                raise ValueError('Excel单表行/列超限，请分表')
            cols=[column(c) for c in range(1,len(fields)+1)]
            digest=hashlib.sha256()
            with z.open(f'xl/worksheets/sheet{i}.xml','w',force_zip64=True) as f:
                f.write((f'<worksheet xmlns="{NS}"><dimension ref="A1:{cols[-1]}{len(rows)+1}"/><sheetViews><sheetView showGridLines="0" workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="15"/><cols>'+''.join(f'<col min="{c}" max="{c}" width="{45 if h=="file" else max(20,min(60,len(str(h))+4))}" customWidth="1"/>' for c,h in enumerate(fields,1))+'</cols><sheetData>').encode())
                values=list(fields)
                digest.update(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()+b'\n')
                f.write(('<row r="1" ht="28" customHeight="1">'+''.join(cell(c+'1',v,True) for c,v in zip(cols,values))+'</row>').encode())
                for n,row in enumerate(rows,2):
                    values=[row.get(h) for h in fields]
                    digest.update(json.dumps(values,ensure_ascii=False,separators=(',',':')).encode()+b'\n')
                    f.write((f'<row r="{n}">'+''.join(cell(c+str(n),v) for c,v in zip(cols,values))+'</row>').encode())
                f.write((f'</sheetData><autoFilter ref="A1:{cols[-1]}{len(rows)+1}"/></worksheet>').encode())
            receipt.append(dict(book=str(path),sheet=name,rows=len(rows),columns=len(fields),typed_sha256=digest.hexdigest()))
            print(f'  XLSX {path.name}/{name}: {len(rows):,} rows',flush=True)
    return receipt
