SELECT EVT_ID, log(NODE_VAL) AS I100
FROM variable_mgt_internal.data.m5_rating_detail
WHERE NODE_NAME = '本期资产估计'
GROUP BY EVT_ID
union 

SELECT EVT_ID, log(NODE_VAL) AS I107
FROM variable_mgt_internal.data.m5_rating_detail
WHERE NODE_NAME = '本期利润总额' or NODE_NAME = '本期财务费用'
GROUP BY EVT_ID;

SELECT EVT_ID, log(NODE_VAL) AS I116
FROM variable_mgt_internal.data.m5_rating_detail
WHERE NODE_NAME = '本期货币资金' or NODE_NAME = '本期_短期资产'
GROUP BY EVT_ID;


SELECT EVT_ID, log(NODE_VAL) AS I116
FROM variable_mgt_internal.data.m5_rating_detail
WHERE NODE_NAME = '本期短期借款' or NODE_NAME = '本期_短期借款'
GROUP BY EVT_ID;
