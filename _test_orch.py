import asyncio, sys, logging, os
logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, '.')

async def test():
    from agents.orchestrator import run_query

    # Test 1: ollama pipeline (default config)
    r1 = await run_query('What is the capital of France?')
    f1 = r1['final']
    print('Test 1 - Policy:', f1['policy'], 'Conf:', int(f1['confidence']*100), '%')
    assert f1['policy'] in ('confident','hedge','re_search'), 'bad policy: ' + str(f1['policy'])

    # Test 2: force mock pipeline
    os.environ['INFERENCE_BACKEND'] = 'mock'
    r2 = await run_query('Who wrote Romeo and Juliet?')
    f2 = r2['final']
    print('Test 2 - Policy:', f2['policy'], 'Backend:', r2.get('inference_backend'))

    print()
    print('All tests passed!')

asyncio.run(test())
