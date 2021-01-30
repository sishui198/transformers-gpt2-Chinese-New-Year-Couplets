from transformers import TFGPT2LMHeadModel, BertTokenizer
import tensorflow as tf
import re
import warnings
warnings.filterwarnings("ignore")

model = TFGPT2LMHeadModel.from_pretrained("gpt2-cn")


def test_model(sentence):
    if " " not in sentence:
        sentence = re.sub("", " ", sentence)[1:]
    # print(sentence)
    tokenizer = BertTokenizer(vocab_file="vocab.txt")
    input_data = tokenizer([sentence], return_tensors="tf",add_special_tokens=False)

    input_ids = input_data["input_ids"][0].numpy()
    input_ids = list(input_ids)
    input_ids.insert(0, tokenizer.get_vocab()["[CLS]"])

    input_ids = tf.constant(input_ids)[None, :]
    # print(input_ids[0].numpy())
    # exit()
    for i in range(100):
        predictions = model(input_ids=input_ids, training=False)[0]

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id[0].numpy(), [3]):
            break

        input_ids = tf.concat([input_ids, predicted_id], axis=-1)
    # print(input_ids)
    result = "".join(tokenizer.batch_decode(tf.squeeze(input_ids, axis=0)))
    result = result.split("|")
    up_sentence = result[0]
    up_sentence = up_sentence.split("]")[1]
    un_sentence = result[1]
    print("上联：", up_sentence)
    print("下联：", un_sentence)


def main():
    while True:
        sentence = str(input("请输入开头："))
        test_model(sentence)


if __name__ == '__main__':
    main()
