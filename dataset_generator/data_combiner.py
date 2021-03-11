import pandas as pd
import json


def detect_objects(result_path):
    object_id_max = 80
    results_json = json.load(open(result_path))
    object_df = pd.DataFrame()

    for result in results_json:
        object_dict = {'filename': result['filename'], 'object_labels': []}

        for i in range(0, object_id_max):
            object_dict[f'object_id_{str(i).rjust(2, "0")}'] = 0

        for detected_object in result['objects']:
            object_dict[f'object_id_{str(detected_object["class_id"]).rjust(2, "0")}'] += 1
            if detected_object['name'] not in object_dict['object_labels']:
                object_dict['object_labels'].append(detected_object['name'])

        object_df = object_df.append(object_dict, ignore_index=True)

    return object_df


def main():
    video_name = "WIZARD OF OZ (1939)"
    movie_df = pd.read_csv(f'temp_imgs\\{video_name}\\movie_data.csv')
    object_df = detect_objects(f'temp_imgs\\{video_name}\\results.json')
    complete_df = pd.concat([movie_df, object_df], axis=1)
    complete_df.to_csv(f'timeline_data\\{video_name}_complete_data.csv', index=False)


if __name__ == "__main__":
    main()
