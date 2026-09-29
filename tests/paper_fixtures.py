"""Deterministic reference-answer fixtures for host-scoring tests, built in memory."""
from flowintentbench.paper_evaluation import PROTOCOL
from flowintentbench.finding_scoring import VERSION as FINDING_PROTOCOL


def reference_review(case):
    ci, meta, gt, material, card = case
    chosen=gt.acceptable_operationalizations[0]
    bid=chosen.operationalization_id
    ops={d.dimension.value:d.statement for d in chosen.decisions}
    lines=['## Operationalization']
    op_lines={}
    for dim,text in ops.items():
        lines.append(text);op_lines[dim]=len(lines)
    lines.append('This choice measures the scientific target using the supplied fields.')
    explanation_line=len(lines)
    lines.append('## Finding')
    findings=[]
    branch=next(b for b in gt.findings_by_operationalization if b.operationalization_id==bid)
    for i,f in enumerate(branch.findings):
        value=f.value
        rendered=repr(value) if not isinstance(value,str) else value
        line=f.statement+': '+rendered+(' '+f.unit if f.unit else '')
        lines.append(line)
        findings.append({'finding_id':f'C{i}','statement':f.statement,'evidence_text':line,
            'source_start_line':len(lines),'source_end_line':len(lines),'value':value,'unit':f.unit,
            'eligible':True,'group_id':'main','mapping_complete':True,'numeric_checks':[],
            'matches':[{'branch_id':bid,'finding_id':f.finding_id,'semantic_verdict':'SUPPORTED','reference_evidence_text':f.statement}]})
    dims=[]
    for dim,text in ops.items():
        dims.append({'dimension':dim,'status':'EXTRACTED','evidence_text':text,
                     'source_start_line':op_lines[dim],'source_end_line':op_lines[dim],
                     'matches':{b.operationalization_id:next(d.statement for d in b.decisions if d.dimension.value==dim)==text for b in gt.acceptable_operationalizations},
                     'adequacy':'MET','reason':'SATISFIES_CONSTRAINT','rationale':'Reference construction fixture.'})
    grades=[{'attribute_id':a['id'],'status':'ASSESSED','grade':4,'source_start_line':op_lines[a['dimension']],
             'source_end_line':op_lines[a['dimension']],'quotation':ops[a['dimension']], 'basis':'ANSWER','reason':'Reference construction fixture.'} for a in card['attributes']]
    explanations=[{'attribute_id':a['id'],'status':'ASSESSED','grade':4,'source_start_line':explanation_line,
             'source_end_line':explanation_line,'quotation':lines[explanation_line-1], 'basis':'ANSWER','reason':'Synthetic structural fixture.'} for a in card['explanation_obligations']]
    review={'protocol':PROTOCOL,'attributes':grades,'explanations':explanations,
            'findings':{'protocol':FINDING_PROTOCOL,'result_groups':[{'group_id':'main','purpose':'PRIMARY',
                       'evidence_text':'\n'.join(lines[1:explanation_line]),'dimensions':dims}],
                       'findings':findings,'extraction_complete':True,'limitations':''}}
    return '\n'.join(lines),review
