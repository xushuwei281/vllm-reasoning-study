"""Make an editable Word copy of the revised report without changing its text."""
from pathlib import Path
import re
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt, RGBColor

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'REPORT.md'
TARGET=PROJECT/'report.docx'


def set_font(style, size, bold=False):
    style.font.name='Arial'
    style.font.size=Pt(size)
    style.font.bold=bold
    style.font.color.rgb=RGBColor.from_string('000000')
    style.font.underline=False
    fonts=style.element.get_or_add_rPr().get_or_add_rFonts()
    for name in ('ascii','hAnsi','eastAsia','cs'):
        fonts.set(qn('w:'+name),'Arial')


def hyperlink(paragraph,label,url):
    link=OxmlElement('w:hyperlink')
    link.set(qn('r:id'),paragraph.part.relate_to(url,RT.HYPERLINK,is_external=True))
    run=OxmlElement('w:r');props=OxmlElement('w:rPr')
    color=OxmlElement('w:color');color.set(qn('w:val'),'176B78');props.append(color)
    underline=OxmlElement('w:u');underline.set(qn('w:val'),'single');props.append(underline)
    run.append(props);text=OxmlElement('w:t');text.text=label;run.append(text)
    link.append(run);paragraph._p.append(link)


def inline(paragraph,text):
    pattern=r'(\*\*.*?\*\*|\[[^\]]+\]\([^)]+\)|`[^`]+`)'
    for token in re.split(pattern,text):
        if not token: continue
        if token.startswith('**') and token.endswith('**'):
            paragraph.add_run(token[2:-2]).bold=True
        elif re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',token):
            match=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',token)
            hyperlink(paragraph,match[1],match[2])
        elif token.startswith('`') and token.endswith('`'):
            paragraph.add_run(token[1:-1]).font.name='Courier New'
        else: paragraph.add_run(token)


def xml_property(parent,name,attributes):
    node=OxmlElement('w:'+name)
    for key,value in attributes.items(): node.set(qn('w:'+key),str(value))
    parent.append(node)
    return node


doc=Document()
section=doc.sections[0]
section.page_width=Inches(8.5);section.page_height=Inches(11)
section.left_margin=section.right_margin=Inches(.75)
section.top_margin=Inches(.65);section.bottom_margin=Inches(.65)
section.header_distance=Inches(.25);section.footer_distance=Inches(.27)
set_font(doc.styles['Normal'],11)
normal=doc.styles['Normal'].paragraph_format
normal.space_after=Pt(7);normal.line_spacing=1.08
normal.widow_control=True
for name,size,after,before in [('Title',22,12,0),('Subtitle',9.5,12,0),
    ('Heading 1',14,7,12),('Heading 2',11.5,6,9),('Caption',9.3,10,4)]:
    style=doc.styles[name]
    set_font(style,size,bold=name in {'Title','Heading 1','Heading 2'})
    style.paragraph_format.space_after=Pt(after)
    style.paragraph_format.space_before=Pt(before)
    style.paragraph_format.line_spacing=1.08
    style.paragraph_format.keep_together=True
    style.paragraph_format.keep_with_next=name in {'Title','Subtitle','Heading 1','Heading 2'}
    for border in list(style.element.xpath('./w:pPr/w:pBdr')):
        border.getparent().remove(border)
doc.styles['Caption'].font.italic=True
doc.styles['Subtitle'].font.italic=False

footer=section.footer.paragraphs[0]
footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.paragraph_format.space_after=Pt(0)
run=footer.add_run();run.font.name='Arial';run.font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE')
field_run=OxmlElement('w:r');field_text=OxmlElement('w:t');field_text.text='1'
field_run.append(field_text);field.append(field_run);run._r.addnext(field)


def add_table(data):
    count=len(data[0])
    weights={2:[1.35,3],3:[2.2,1,1.2],4:[1.3,1,.9,1],5:[1.35,1,1,1.15,1.2]}[count]
    widths=[Inches(7*w/sum(weights)) for w in weights]
    table=doc.add_table(rows=len(data),cols=count)
    table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
    for column,width in zip(table.columns,widths): column.width=width
    props=table._tbl.tblPr
    borders=xml_property(props,'tblBorders',{})
    for edge in ('top','left','bottom','right','insideH','insideV'):
        xml_property(borders,edge,{'val':'single','sz':'4','color':'D9D9D9'})
    margins=xml_property(props,'tblCellMar',{})
    for edge,value in (('top',85),('bottom',85),('left',105),('right',105)):
        xml_property(margins,edge,{'w':value,'type':'dxa'})
    for row_index,(row,values) in enumerate(zip(table.rows,data)):
        trpr=row._tr.get_or_add_trPr()
        xml_property(trpr,'cantSplit',{})
        if row_index==0: xml_property(trpr,'tblHeader',{'val':'true'})
        for column,(cell,value,width) in enumerate(zip(row.cells,values,widths)):
            cell.width=width;cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell_props=cell._tc.get_or_add_tcPr()
            shade='DFEDF1' if row_index==0 else ('F6F8FA' if row_index%2==0 else 'FFFFFF')
            xml_property(cell_props,'shd',{'fill':shade,'val':'clear'})
            p=cell.paragraphs[0]
            p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(0)
            p.paragraph_format.line_spacing=1.05
            p.paragraph_format.keep_together=True
            p.paragraph_format.keep_with_next=row_index<len(data)-1
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if column==0 or count==2 else WD_ALIGN_PARAGRAPH.CENTER
            inline(p,value)
            for run in p.runs:
                run.font.name='Arial';run.font.size=Pt(9.3)
                run.font.color.rgb=RGBColor(0,0,0)
                if row_index==0:run.bold=True
    spacer=doc.add_paragraph()
    spacer.paragraph_format.space_after=Pt(2)
    spacer.paragraph_format.space_before=Pt(0)
    spacer.paragraph_format.line_spacing=1
    spacer.add_run().font.size=Pt(2)


lines=SOURCE.read_text().splitlines();i=0;figures=0;table_count=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('# '):
        p=doc.add_paragraph(style='Title');inline(p,line[2:]);i+=1;continue
    if line.startswith('## '):
        p=doc.add_paragraph(style='Heading 1');inline(p,line[3:]);i+=1;continue
    if line.startswith('### '):
        p=doc.add_paragraph(style='Heading 2');inline(p,line[4:]);i+=1;continue
    if line.startswith('!['):
        match=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_together=True;p.paragraph_format.keep_with_next=True
        p.paragraph_format.space_before=Pt(5);p.paragraph_format.space_after=Pt(0)
        picture=p.add_run().add_picture(str(PROJECT/match[2]),width=Inches(7))
        picture._inline.docPr.set('descr',match[1])
        caption=doc.add_paragraph(style='Caption');inline(caption,match[1])
        figures+=1;i+=1;continue
    if line.startswith('|'):
        data=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',x) for x in cells):data.append(cells)
            i+=1
        add_table(data);table_count+=1;continue
    parts=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
        parts.append(lines[i].strip());i+=1
    content=' '.join(parts)
    style='Subtitle' if content.startswith('**Experiments:') else 'Normal'
    p=doc.add_paragraph(style=style);p.paragraph_format.keep_together=True
    if re.match(r'^\[[1-4]\] ',content):p.paragraph_format.keep_with_next=True
    inline(p,content)

doc.core_properties.title='Tuning vLLM for Small-Model Reasoning Under a Fixed GPU-Time Budget'
doc.core_properties.subject='Revised experimental report with editable text and tables'
doc.core_properties.author=''
doc.core_properties.last_modified_by=''
doc.core_properties.comments=''
doc.save(TARGET)
print(TARGET)
print(f'{table_count} native tables; {figures} embedded figures')
