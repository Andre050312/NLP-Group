"""Build the Word review copy using the managed document runtime and Pillow.

Run with the bundled Python documented in docs/report_review/README.md.
No network requests or trained models are used. Input prose and tables remain
editable in docs/report_draft.md; numeric plots use recorded_results.json.
"""
import json
from pathlib import Path
import re
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/report_review/figures"
BLUE, ORANGE, INK, GRID = "#176599", "#BA5D18", "#17212B", "#D9D9D9"


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def label(draw, xy, text, size=30, color=INK, bold=False, anchor=None):
    draw.text(xy, text, font=font(size, bold), fill=color, anchor=anchor)


def figures():
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "docs/report_review/recorded_results.json").read_text())
    im = Image.new("RGB", (2000, 690), "white")
    d = ImageDraw.Draw(im)
    def box(rect, title, subtitle, fill):
        d.rounded_rectangle(rect, radius=16, fill=fill, outline=GRID, width=3)
        x0, y0, x1, y1 = rect
        label(d, ((x0+x1)/2, y0+25), title, 34, bold=True, anchor="mt")
        label(d, ((x0+x1)/2, y0+76), subtitle, 27, anchor="mt")
    def arrow(x0, y0, x1, y1):
        d.line((x0, y0, x1, y1), fill="#68737D", width=5)
        if y0 == y1:
            d.polygon([(x1,y1),(x1-15,y1-10),(x1-15,y1+10)], fill="#68737D")
        else:
            d.polygon([(x1,y1),(x1-10,y1-15),(x1+10,y1-15)], fill="#68737D")
    label(d, (45, 20), "Aspect-conditioned input", 34, bold=True)
    box((45,85,650,215), "Sentence + gold target", "paired with the aspect query", "#F0F3F5")
    box((45,300,650,440), "DeBERTaV3 encoder", "shared contextual attention", "#EAF3FA")
    arrow(350,215,350,300)
    for top, title, subtitle, fill in [(35,"CLS vector","final feature removed in no CLS", "#FCF1E8"),
                                       (235,"Target mean","gold span in the context", "#EAF3FA"),
                                       (435,"Query mean","aspect in the second sequence", "#EAF3FA")]:
        box((800,top,1400,top+135),title,subtitle,fill)
    d.line((650,370,720,370,720,102,800,102), fill="#68737D", width=5)
    arrow(720,302,800,302)
    d.line((720,370,720,502,800,502), fill="#68737D", width=5)
    d.line((1400,102,1460,102,1460,502,1400,502), fill="#68737D", width=5)
    arrow(1460,302,1540,302)
    box((1540,235,1950,370),"MLP head","3 class probabilities", "#F0F3F5")
    label(d,(45,610),"No CLS changes the final feature concatenation, not encoder attention or sentence access.",30)
    im.save(ASSETS / "architecture.png", dpi=(300,300))

    im = Image.new("RGB", (2000, 850), "white"); d = ImageDraw.Draw(im)
    left, right, top, bottom = 175, 1900, 105, 665
    lo, hi = .77, .84
    yp = lambda value: bottom-(value-lo)/(hi-lo)*(bottom-top)
    label(d,(30,25),"Macro-F1 on the official test",36,bold=True)
    for value in [.77,.78,.79,.80,.81,.82,.83,.84]:
        y=yp(value); d.line((left,y,right,y),fill=GRID,width=2)
        label(d,(left-22,y),f"{value:.2f}",30,anchor="rm")
    for idx, seed in enumerate(data["seeds"]):
        x=left+150+idx*340
        a,b=data["control_macro"][idx],data["no_cls_macro"][idx]
        d.line((x-42,yp(a),x+42,yp(b)),fill="#AAB2B9",width=6)
        d.ellipse((x-54,yp(a)-12,x-30,yp(a)+12),fill=BLUE)
        d.rectangle((x+30,yp(b)-12,x+54,yp(b)+12),fill=ORANGE)
        label(d,(x-45,yp(a)+24),f"{a:.4f}",27,BLUE,anchor="mt")
        label(d,(x+45,yp(b)-27),f"{b:.4f}",27,ORANGE,anchor="mb")
        label(d,(x,bottom+42),str(seed),32,anchor="mt")
    label(d,(920,780),"Training seed",30,anchor="mm")
    d.ellipse((1350,28,1374,52),fill=BLUE);label(d,(1390,22),"Control",30,BLUE)
    d.rectangle((1630,28,1654,52),fill=ORANGE);label(d,(1670,22),"No CLS",30,ORANGE)
    im.save(ASSETS / "seed_pairs.png",dpi=(300,300))

    im=Image.new("RGB",(2000,800),"white"); d=ImageDraw.Draw(im)
    left,right,top,bottom=635,1850,130,650
    xp=lambda v:left+(v+.006)/.018*(right-left)
    label(d,(35,25),"Change in Macro-F1 after validation-selected offset",36,bold=True)
    for value in [-.004,0,.004,.008,.012]:
        x=xp(value);d.line((x,top,x,bottom),fill="#68737D" if value==0 else GRID,width=3 if value==0 else 2)
        label(d,(x,bottom+30),f"{value:+.3f}",29,anchor="mt")
    names=[("Fixed split control","fixed_control",BLUE),("Fixed split no CLS","fixed_no_cls",ORANGE),
           ("Outer fold control","outer_control",BLUE),("Outer fold no CLS","outer_no_cls",ORANGE)]
    for idx,(title,key,color) in enumerate(names):
        y=190+idx*125; before,after=data["macro_summary"][key]; delta=after-before
        label(d,(left-35,y),title,32,anchor="rm")
        d.rectangle((min(xp(0),xp(delta)),y-25,max(xp(0),xp(delta)),y+25),fill=color)
        label(d,(xp(delta)+(18 if delta>=0 else -18),y),f"{delta:+.4f}",31,color,anchor="lm" if delta>=0 else "rm")
    label(d,(1200,755),"Difference in Macro-F1",30,anchor="mm")
    im.save(ASSETS / "offset_effects.png",dpi=(300,300))


def style_run(run, size=None, bold=None, color="000000"):
    run.font.name="Times New Roman"
    run.font.color.rgb=RGBColor.from_string(color)
    if size: run.font.size=Pt(size)
    if bold is not None: run.bold=bold


def make_table(doc, lines):
    cells=[[x.strip() for x in line.strip().strip("|").split("|")] for line in lines]
    cells=[row for row in cells if not all(re.fullmatch(r":?-+:?",c) for c in row)]
    n=len(cells[0]); table=doc.add_table(rows=0, cols=n)
    table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=False
    widths={3:[1.7,2.65,2.65],4:[2.15,1.12,1.25,2.48],5:[2.15,1.21,1.21,1.21,1.22],6:[1.40,1.16,1.16,1.00,1.00,1.28]}[n]
    # Available width is 7 inches on Letter with 0.75 inch margins.
    for col,w in zip(table.columns,widths):col.width=Inches(w)
    for i,row in enumerate(cells):
        out=table.add_row()
        trPr=out._tr.get_or_add_trPr()
        cant=OxmlElement("w:cantSplit");trPr.append(cant)
        if i==0:
            repeat=OxmlElement("w:tblHeader");trPr.append(repeat)
        for j,(cell,text) in enumerate(zip(out.cells,row)):
            cell.width=Inches(widths[j]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            props=cell._tc.get_or_add_tcPr()
            shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"263746" if i==0 else ("F2F5F7" if i%2 else "FFFFFF"));props.append(shade)
            margins=OxmlElement("w:tcMar")
            for side in ("top","left","bottom","right"):
                el=OxmlElement("w:"+side);el.set(qn("w:w"),"75");el.set(qn("w:type"),"dxa");margins.append(el)
            props.append(margins)
            borders=OxmlElement("w:tcBorders")
            for side in ("top","left","bottom","right"):
                el=OxmlElement("w:"+side);el.set(qn("w:val"),"single");el.set(qn("w:sz"),"4");el.set(qn("w:color"),"D9D9D9");borders.append(el)
            props.append(borders)
            p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1.05
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 or n==3 else WD_ALIGN_PARAGRAPH.CENTER
            # Let long uncertainty entries wrap at ±, but keep each number whole.
            text=text.replace(" ± ","\n± ") if n==6 else text
            style_run(p.add_run(text),9.4,i==0,"FFFFFF" if i==0 else "000000")
    doc.add_paragraph().paragraph_format.space_after=Pt(1)


def native_offset_equation(doc):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    para=OxmlElement("m:oMathPara");math=OxmlElement("m:oMath");para.append(math)
    def atom(text):
        r=OxmlElement("m:r");t=OxmlElement("m:t");t.text=text;r.append(t);return r
    def sub(base,index):
        s=OxmlElement("m:sSub");e=OxmlElement("m:e");e.append(atom(base));v=OxmlElement("m:sub");v.append(atom(index));s.extend((e,v));return s
    math.append(sub("p′","neutral"));math.append(atom(" = "))
    frac=OxmlElement("m:f");num=OxmlElement("m:num");num.append(atom("k "));num.append(sub("p","neutral"))
    den=OxmlElement("m:den")
    for item in (sub("p","negative"),atom(" + k "),sub("p","neutral"),atom(" + "),sub("p","positive")):den.append(item)
    frac.extend((num,den));math.append(frac);p._p.append(para)


def main():
    figures();doc=Document();s=doc.sections[0]
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=s.bottom_margin=s.left_margin=s.right_margin=Inches(.75)
    s.header_distance=s.footer_distance=Inches(.3)
    for name in ("Normal","Title","Subtitle","Heading 1","Heading 2","Heading 3","Caption"):
        st=doc.styles[name];st.font.name="Times New Roman";st.font.color.rgb=RGBColor(0,0,0)
        fonts=st.element.get_or_add_rPr().find(qn("w:rFonts"))
        for key in list(fonts.attrib):
            if "theme" in key.lower(): del fonts.attrib[key]
        fonts.set(qn("w:eastAsia"),"Times New Roman");fonts.set(qn("w:cs"),"Times New Roman")
        props=st.element.find(qn("w:pPr"))
        if props is not None:
            for border in list(props.findall(qn("w:pBdr"))):props.remove(border)
    doc.styles["Caption"].font.bold=False
    norm=doc.styles["Normal"];norm.font.size=Pt(11)
    norm.paragraph_format.line_spacing=1.08;norm.paragraph_format.space_after=Pt(6)
    for name,size in (("Title",20),("Heading 1",14),("Heading 2",11.5)):
        st=doc.styles[name];st.font.size=Pt(size);st.font.bold=True
        st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(6)
    doc.styles["Title"].paragraph_format.space_before=Pt(0)
    hp=s.header.paragraphs[0];hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    style_run(hp.add_run("COMP4136  |  Group 9  |  Neutral aspect sentiment"),9)
    fp=s.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    style_run(fp.add_run("Group 9  •  Review copy  •  "),9)
    fld=OxmlElement("w:fldSimple");fld.set(qn("w:instr"),"PAGE");fp._p.append(fld)
    source=(ROOT/"docs/report_draft.md").read_text(encoding="utf-8").splitlines()
    i=0;in_refs=False
    while i<len(source):
        line=source[i].strip()
        if not line:i+=1;continue
        if line=="<!-- PAGEBREAK -->":doc.add_page_break();i+=1;continue
        if line.startswith("|"):
            rows=[]
            while i<len(source) and source[i].strip().startswith("|"):
                rows.append(source[i]);i+=1
            make_table(doc,rows);continue
        if line.startswith("!["):
            path=re.search(r"\]\((.+)\)",line).group(1)
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next=True
            p.add_run().add_picture(str(ROOT/"docs"/path),width=Inches(6.85));i+=1;continue
        if line.startswith("# "):
            doc.add_paragraph(line[2:],"Title");i+=1;continue
        if line.startswith("## "):
            title=line[3:];in_refs=title=="References";doc.add_heading(title,1);i+=1;continue
        if line.startswith("### "):
            doc.add_heading(line[4:],2);i+=1;continue
        if line.startswith(("Figure ","Table ")):
            p=doc.add_paragraph(line,"Caption");p.paragraph_format.space_after=Pt(7)
            p.paragraph_format.keep_with_next=line.startswith("Table ")
            for run in p.runs:style_run(run,9.5)
        else:
            p=doc.add_paragraph(line)
            if in_refs:
                p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(7)
                p.paragraph_format.left_indent=Inches(.26);p.paragraph_format.first_line_indent=Inches(-.26)
                for run in p.runs:style_run(run,10)
            elif line.startswith("Model Ablations"):
                for run in p.runs:style_run(run,12)
            elif line.startswith(("COMP4136 ","Zhang Xuanrui")):
                for run in p.runs:style_run(run,10)
            if line.startswith("For a positive scalar k"):
                native_offset_equation(doc)
        i+=1
    doc.core_properties.title="Diagnosing Neutral Sentiment in Aspect Based Sentiment Analysis"
    doc.core_properties.author="Group 9 — Zhang Xuanrui; Zhao Yuzhou; Hua Chunyang"
    doc.core_properties.subject="COMP4136 group project review copy"
    doc.core_properties.comments="Built from tracked report source. Experiment scores are group-recorded, not rerun."
    out=ROOT/"reports/Group_9_ABSA_Report_Review.docx";out.parent.mkdir(exist_ok=True)
    doc.save(out);print(out)


if __name__=="__main__":main()
