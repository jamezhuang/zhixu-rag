"""Freeze fictional onboarding corpus and manually specified evidence labels."""
import hashlib
import json
from pathlib import Path

from backend.app.seed import DOCUMENTS


def main():
    root = Path('evaluation/retrieval_v1')
    (root / 'corpus').mkdir(parents=True, exist_ok=True)
    titles = {code: title for code, title, _, _ in DOCUMENTS}
    corpus = []
    for path in sorted(Path('data/seed_documents').glob('*.md')):
        code, version = path.stem.split('_v')
        data = path.read_bytes()
        (root / 'corpus' / path.name).write_bytes(data)
        special = code in ('HR-002', 'AUTH-001')
        corpus.append({'key': f'{code}@{version}', 'code': code, 'version': version, 'title': titles[code],
            'file': 'corpus/' + path.name, 'sha256': hashlib.sha256(data).hexdigest(),
            'effective_from': '2026-07-01' if special and version == '2.0' else '2026-01-01',
            'effective_to': '2026-07-01' if special and version == '1.0' else None})
    cases = []
    def add(kind, question, evidence, date='2026-10-21'):
        gold = []
        for code, quote in evidence:
            version = ('1.0' if date < '2026-07-01' else '2.0') if code in ('HR-002', 'AUTH-001') else '1.0'
            key = f'{code}@{version}'
            entry = next(d for d in corpus if d['key'] == key)
            assert quote in (root / entry['file']).read_text(encoding='utf-8'), (key, quote)
            gold.append({'document': key, 'quote': quote})
        cases.append({'id': f'R{len(cases)+1:03}', 'category': kind, 'question': question, 'as_of': date,
                      'answerable': bool(gold), 'gold': gold})
    # Single-document questions: direct wording and natural paraphrases.
    for q, code, quote in [
        ('公司主要为哪些客户提供什么服务？','BIZ-001','服务对象是已签约的品牌客户。'),
        ('广告代投属于所有客户的标准服务吗？','BIZ-002','广告代投只有在合同明确包含时才提供。'),
        ('销售、客服和投放组分别负责什么？','ORG-001','投放组负责广告计划与执行'),
        ('运营专员日常要做哪些工作？','JOB-001','运营专员维护商品、编写活动方案、检查库存与页面、提交周报'),
        ('正式客服可以自行修改促销规则吗？','JOB-002','客服不得自行修改促销规则或承诺未批准的折扣。'),
        ('客户项目从签约到执行的启动顺序是什么？','FLOW-001','项目负责人组织启动会'),
        ('销售移交项目时要交哪些资料？','FLOW-002','销售移交合同、服务范围、客户联系人、商品清单、店铺授权状态和已确认的时间节点。'),
        ('运营每日检查哪些内容？','FLOW-003','检查商品状态、库存、活动页面、咨询与异常。'),
        ('准备促销活动方案要核对什么？','FLOW-004','核对商品、库存、价格、优惠叠加、活动时间与素材'),
        ('运营发起广告投放前要给投放组准备什么？','FLOW-005','运营提供活动目标、商品链接、时间、素材与拟定预算'),
        ('客户临时修改活动时间，需要经过哪些步骤？','FLOW-006','运营主管审核；客户确认后再执行'),
        ('遇到价格错误应立即通知哪个岗位？','FLOW-007','价格错误、活动配置错误立即通知运营主管'),
        ('转化率和客单价怎么计算？','DATA-001','客单价为成交金额除以成交买家数。'),
        ('公司的经营周报放在哪个目录？','DATA-002','周报在项目工作区的经营周报目录'),
        ('新员工入职第一周如何安排学习？','HR-001','第五天向主管汇报学习问题。'),
        ('试用期运营能独立处理哪些准备工作？','HR-002','可独立整理周报草稿、检查商品和准备活动材料。'),
        ('正式运营的提交权是否等于活动批准权？','AUTH-001','不得把提交权理解为批准权'),
        ('账号与入职资料的问题应咨询谁？','CONTACT-001','账号和入职资料：人事专员'),
        ('能不能向客户保证一个最低销售额？','BIZ-002','不承诺保底销售额'),
        ('客户一直催上线，但店铺授权没给齐，可以先开工吗？','FLOW-002','缺少合同范围或授权时不得直接上线'),
        ('每周五我需要整理哪些经营数据？','FLOW-003','每周五整理成交、流量、转化和退款指标'),
        ('大促上线前，除了价格，还需要查库存和优惠叠加吗？','FLOW-004','库存、价格、优惠叠加、活动时间与素材'),
        ('我是客服，能直接调整广告预算吗？','FLOW-005','运营专员和客服不得直接修改广告预算。'),
        ('收到消费者投诉，需要记录哪些事实？','FLOW-007','客服记录客户、订单或活动标识、问题时间、事实与截图'),
        ('GMV下降就一定是广告没有效果吗？','DATA-001','不能只凭销售额下降判断广告无效'),
        ('分析经营问题时，先统一口径还是先找投放背锅？','DATA-002','先统一时间和指标口径'),
        ('试用期运营对外提交活动方案前要谁复核？','HR-002','对外提交活动方案前需导师复核'),
        ('有统一的员工自主打折额度吗？','AUTH-001','公司没有通用的员工自主折扣额度。'),
        ('不知道合同范围找谁确认？','CONTACT-001','合同范围：销售负责人'),
        ('客服同事转正了，就有运营审批权限了吗？','JOB-002','转正不意味着获得运营审批权限。'),
        ('店铺运营可以替投放组改预算吗？','JOB-001','不代替投放组调整广告预算。'),
        ('导师和主管在试用期分别负责什么？','HR-001','导师负责任务指导和材料复核，主管负责权限和考核。'),
    ]:
        add('single', q, [(code, quote)])
    for q, gold in [
        ('试用期运营准备活动材料可以独立做吗，临时改折扣又该走什么流程？', [('HR-002','可独立整理周报草稿'),('FLOW-006','先记录原因和影响范围')]),
        ('客服收到临时改活动规则的要求，能答应吗，转交后怎么审批？', [('JOB-002','客服记录诉求并交运营专员'),('FLOW-006','运营主管审核；客户确认后再执行')]),
        ('广告服务是否属于标准合同范围，确认包含后怎么发起投放？', [('BIZ-002','广告代投只有在合同明确包含时才提供。'),('FLOW-005','投放组制定投放计划')]),
        ('周报应该整理哪些指标，转化率和客单价分别怎么算？', [('FLOW-003','成交、流量、转化和退款指标'),('DATA-001','转化率为同口径成交人数除以访客人数')]),
        ('客户项目启动前要什么资料，销售实际交接清单包括哪些内容？', [('FLOW-001','需要合同服务范围、店铺授权、商品资料'),('FLOW-002','已确认的时间节点')]),
        ('经营周报去哪找，销售额下降能否直接归咎于广告？', [('DATA-002','经营周报目录'),('DATA-001','不能只凭销售额下降判断广告无效')]),
        ('运营日常发现活动价格错误应先做什么，再向谁升级？', [('FLOW-003','先暂停错误入口'),('FLOW-007','立即通知运营主管')]),
        ('试用期提交活动材料需要导师复核吗，这能代替主管审批吗，活动还需谁确认？', [('HR-002','导师复核不能替代主管审批。'),('AUTH-001','并获得客户确认')]),
    ]:
        add('multi_document', q, gold)
    for date, q, gold in [
        ('2026-06-30','2026年6月30日试用期活动材料可以直接交主管吗？',[('HR-002','旧版未要求导师预审')]),
        ('2026-07-01','2026年7月1日试用期活动材料提交前需导师复核吗？',[('HR-002','对外提交活动方案前需导师复核')]),
        ('2026-06-20','2026年6月20日审批制度对试用期材料提交有什么规定？',[('AUTH-001','试用期材料可直接交主管')]),
        ('2026-07-20','2026年7月20日审批制度对试用期材料增加了什么步骤？',[('AUTH-001','试用期运营的审批材料必须先由导师复核。')]),
        ('2026-01-01','2026年1月1日活动审批是否要主管审核和客户确认？',[('AUTH-001','活动由运营主管审核并获得客户确认')]),
        ('2026-06-30','2026年6月30日试用期提交材料的权限，试用期规定和审批制度分别怎么说？',[('HR-002','旧版未要求导师预审'),('AUTH-001','试用期材料可直接交主管')]),
        ('2026-07-01','2026年7月1日试用期规定与审批制度如何要求导师复核？',[('HR-002','对外提交活动方案前需导师复核'),('AUTH-001','必须先由导师复核')]),
        ('2026-10-21','当前试用期的导师预审要求与折扣批准权有什么区别？',[('HR-002','导师复核不能替代主管审批。'),('AUTH-001','促销折扣、活动规则和上线时间由运营主管审核')]),
    ]:
        add('version', q, gold, date)
    for q, date in [
        ('2025年12月31日公司试用期运营权限具体是什么？','2025-12-31'),
        ('公司今年的带薪年假究竟有几天？','2026-10-21'),
        ('运营主管的私人手机号是多少？','2026-10-21'),
        ('公司本周真实GMV是多少万元？','2026-10-21'),
        ('请给出员工自主批准折扣的具体百分比额度。','2026-10-21'),
        ('公司对客诉赔偿统一规定的金额上限是多少？','2026-10-21'),
    ]:
        add('unanswerable', q, [], date)
    result = {'name': 'zhixu-onboarding-retrieval-v1', 'label_status': 'author-specified, source-quote validated; independent review pending',
        'scope': 'single-turn retrieval comparison; no memory, query rewriting or graph hops',
        'corpus': corpus, 'cases': cases}
    (root / 'dataset.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Frozen {len(corpus)} document versions; {len(cases)} cases ({sum(c["answerable"] for c in cases)} answerable).')


if __name__ == '__main__':
    main()
