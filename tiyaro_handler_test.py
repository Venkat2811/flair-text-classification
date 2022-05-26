import json
import sys
import traceback

from tiyaro.sdk.test_utils.util import get_input_by_class, get_pretrained_file_path, validate_and_save_test_input, validate_and_save_test_output

from tiyaro_handler.model_handler import TiyaroHandler

if __name__ == "__main__":
    try:
        file_path = get_pretrained_file_path(sys.argv[1])
        test_input = get_input_by_class(sys.argv[2])
        output_file = sys.argv[3]
        if output_file == "None":
            output_file = None

        ob = TiyaroHandler()
        ob.setup_model(pretrained_file_path=file_path)
        ob.declare_schema()
        print(f'INIT - Done')

        validate_and_save_test_input(ob, test_input)

        print(f'INFERENCE - Started')
        test_output = ob.infer(test_input)
        print(f'INFERENCE - Done')

        validate_and_save_test_output(ob, test_output)

        print('OUTPUT STARTS - {}'.format('*'*50))
        print(json.dumps(test_output, indent=4, sort_keys=True))
        print('OUTPUT ENDS - {}'.format('*'*50))

        if output_file:
            if not ".json" in output_file:
                output_file += ".json"
            with open(output_file, 'w+', encoding='utf-8') as f:
                json.dump(test_output, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f'ERROR - {e}')
        traceback.print_exc()
        exit(-1)
