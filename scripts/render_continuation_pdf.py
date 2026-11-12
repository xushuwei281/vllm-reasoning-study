"""Render REPORT.md as a readable, linked research PDF using ReportLab."""
from pathlib import Path
import re
import html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

root=Path(__file__).resolve().parents[1]
source=root/'REPORT.md';target=root/'report.pdf'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyResearch',fontName='Helvetica',fontSize=10.1,leading=14.1,spaceAfter=8,allowWidows=0,allowOrphans=0,textColor=colors.HexColor('#263447')))
styles.add(ParagraphStyle(name='ReportTitle',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=colors.HexColor('#12324b'),spaceAfter=15))
styles.add(ParagraphStyle(name='SectionResearch',fontName='Helvetica-Bold',fontSize=13.2,leading=17,textColor=colors.HexColor('#176b78'),spaceBefore=13,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='SubsectionResearch',fontName='Helvetica-Bold',fontSize=10.5,leading=14,textColor=colors.HexColor('#176b78'),spaceBefore=9,spaceAfter=6,keepWithNext=True))
styles.add(ParagraphStyle(name='CellResearch',fontName='Helvetica',fontSize=8.3,leading=11.2,spaceAfter=0))
styles.add(ParagraphStyle(name='CellHeaderResearch',parent=styles['CellResearch'],fontName='Helvetica-Bold',textColor=colors.HexColor('#12324b')))
styles.add(ParagraphStyle(name='CaptionResearch',fontName='Helvetica-Oblique',fontSize=8,leading=11,spaceAfter=10,textColor=colors.HexColor('#48566a')))
styles.add(ParagraphStyle(name='CodeResearch',fontName='Courier',fontSize=7.7,leading=11,spaceAfter=9,backColor=colors.HexColor('#f3f6f8'),borderPadding=6))

# Supported subset is deliberately small; report generation controls the Markdown.
def inline(text):
    tokens={}
    def token(value):
        key=f'ZZTOKEN{len(tokens)}ZZ';tokens[key]=value;return key
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:token(f'<link href="{html.escape(m[2],quote=True)}" color="#176b78">{html.escape(m[1])}</link>'),text)
    text=re.sub(r'`([^`]+)`',lambda m:token('<font name="Courier" size="8.3">'+html.escape(m[1])+'</font>'),text)
    text=html.escape(text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    for key,value in tokens.items():text=text.replace(key,value)
    return text

def header(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setStrokeColor(colors.HexColor('#c8d7df'));canvas.line(45,h-33,w-45,h-33)
    canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor('#526372'))
    canvas.drawString(45,h-25,'vLLM / SMALL-MODEL REASONING / EXPERIMENTAL STUDY')
    canvas.drawRightString(w-45,h-25,'REVISED 4 NOVEMBER 2026')
    canvas.drawString(45,25,'One GPU | Live optimizer comparison | Held-out reasoning test')
    canvas.drawRightString(w-45,25,str(doc.page));canvas.restoreState()

story=[];lines=source.read_text().splitlines();i=0;width=A4[0]-90
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['ReportTitle']));i+=1;continue
    if line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['SubsectionResearch']));i+=1;continue
    if line.startswith('## '):story.append(Paragraph(inline(line[3:]),styles['SectionResearch']));i+=1;continue
    if line.startswith('!['):
        m=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line);p=root/m[2];w,h=ImageReader(str(p)).getSize();scale=min(width/w,255/h)
        story.append(KeepTogether([Image(str(p),w*scale,h*scale),Spacer(1,5),Paragraph(inline(m[1]),styles['CaptionResearch'])]));i+=1;continue
    if line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in cells):table.append(cells)
            i+=1
        cols=len(table[0]);weights={2:[1.35,3],3:[2.2,1,1.2],4:[1.3,1,.9,1],5:[1.35,1,1,1.15,1.2]}.get(cols,[1]*cols)
        if len(weights)!=cols:weights=[1]*cols
        widths=[width*w/sum(weights) for w in weights]
        body=[[Paragraph(inline(c),styles['CellHeaderResearch' if n==0 else 'CellResearch']) for c in row] for n,row in enumerate(table)]
        flow=Table(body,colWidths=widths,repeatRows=1,hAlign='LEFT')
        flow.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfedf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7598a5')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dfe4')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fa')])]))
        story.append(KeepTogether([flow,Spacer(1,11)]));continue
    paragraph=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
        paragraph.append(lines[i].strip());i+=1
    text=' '.join(paragraph)
    style=styles['CodeResearch'] if text.startswith('`') and text.endswith('`') else styles['BodyResearch']
    story.append(KeepTogether([Paragraph(inline(text),style)]))

# Keep each heading with the complete first paragraph, including consecutive
# section/subsection headings. A wrapped paragraph alone can otherwise leave
# a heading behind when ReportLab moves that paragraph to the following page.
grouped=[];headings=[]
for flow in story:
    if isinstance(flow,Paragraph) and flow.style.name in {'ReportTitle','SectionResearch','SubsectionResearch'}:
        headings.append(flow)
    elif headings:
        following=flow._content if isinstance(flow,KeepTogether) else [flow]
        grouped.append(KeepTogether(headings+following));headings=[]
    else:grouped.append(flow)
grouped.extend(headings)

doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=47,bottomMargin=45,title='Tuning vLLM for Small-Model Reasoning Under a Fixed GPU-Time Budget',author='vLLM reasoning study',pageCompression=1)
doc.build(grouped,onFirstPage=header,onLaterPages=header)
print(target)
