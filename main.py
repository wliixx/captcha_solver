import argparse
import os
import random
import string
 
import numpy as np
import torch
import torch.nn as nn
from captcha.image import ImageCaptcha
from PIL import Image
from torch.utils.data import DataLoader, Dataset
 
# ----------------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------------
CHARS = string.digits + string.ascii_lowercase   # алфавит капчи (36 символов)
CAPTCHA_LEN = 5                                  # длина текста
H, W = 60, 160                                   # размер картинки
BLANK = 0                                        # индекс "пустого" символа CTC
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 занят под blank
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(CHARS) + 1
MODEL_PATH = "captcha_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
 
# ----------------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------------
def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL-картинка -> тензор (1, H, W) со значениями 0..1 (оттенки серого)."""
    img = img.convert("L").resize((W, H))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)
 
 
class CaptchaDataset(Dataset):
    """Каждый элемент - свежесгенерированная капча и её текст.
 
    fixed=True делает набор детерминированным (для валидации): так мы
    сравниваем качество модели на одних и тех же картинках после каждой эпохи.
    """
 
    def __init__(self, size: int, fixed: bool = False):
        self.size = size
        self.fixed = fixed
        self.generator = None  # создаётся лениво (удобно для DataLoader workers)
 
    def __len__(self):
        return self.size
 
    def __getitem__(self, i):
        if self.generator is None:
            self.generator = ImageCaptcha(width=W, height=H)
        rng = random.Random(i) if self.fixed else random
        text = "".join(rng.choices(CHARS, k=CAPTCHA_LEN))
        img = self.generator.generate_image(text)
        target = torch.tensor([CHAR2IDX[c] for c in text], dtype=torch.long)
        return image_to_tensor(img), target
 
 
# ----------------------------------------------------------------------------
# Модель
# ----------------------------------------------------------------------------
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
 
        def block(cin, cout, pool):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )
 
        # Сжимаем картинку по высоте сильнее, чем по ширине:
        # ширина превращается в "время" (последовательность столбцов).
        self.cnn = nn.Sequential(
            block(1, 32, (2, 2)),     # 60x160 -> 30x80
            block(32, 64, (2, 2)),    # 30x80  -> 15x40
            block(64, 128, (2, 1)),   # 15x40  -> 7x40
            block(128, 256, (2, 1)),  # 7x40   -> 3x40
        )
        self.proj = nn.Linear(256 * 3, 128)
        self.rnn = nn.GRU(128, 128, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256, NUM_CLASSES)
 
    def forward(self, x):
        f = self.cnn(x)                                   # (B, 256, 3, 40)
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)    # (B, 40, 768)
        f = torch.relu(self.proj(f))                      # (B, 40, 128)
        f, _ = self.rnn(f)                                # (B, 40, 256)
        return self.fc(f)                                 # (B, 40, классы)
 
 
def greedy_decode(logits: torch.Tensor):
    """Жадное декодирование CTC: берём лучший класс на каждом шаге,
    схлопываем повторы подряд и выкидываем blank."""
    best = logits.argmax(dim=-1).cpu().numpy()            # (B, T)
    results = []
    for seq in best:
        out, prev = [], BLANK
        for k in seq:
            if k != prev and k != BLANK:
                out.append(IDX2CHAR[int(k)])
            prev = k
        results.append("".join(out))
    return results
 
 
# ----------------------------------------------------------------------------
# Обучение
# ----------------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = greedy_decode(model(x.to(DEVICE)))
            for p, t in zip(preds, y):
                truth = "".join(IDX2CHAR[int(i)] for i in t)
                correct += p == truth
                total += 1
    return correct / total
 
 
def train(epochs: int, samples_per_epoch: int, batch_size: int, lr: float):
    workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(CaptchaDataset(samples_per_epoch),
                              batch_size=batch_size, num_workers=workers)
    val_loader = DataLoader(CaptchaDataset(2000, fixed=True),
                            batch_size=256, num_workers=workers)
 
    model = CRNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    best_acc = 0.0
    print(f"Устройство: {DEVICE}")
 
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)                                   # (B, T, C)
            log_probs = logits.log_softmax(-1).permute(1, 0, 2)  # (T, B, C)
            in_lens = torch.full((x.size(0),), log_probs.size(0), dtype=torch.long)
            tgt_lens = torch.full((x.size(0),), CAPTCHA_LEN, dtype=torch.long)
            loss = ctc(log_probs, y, in_lens, tgt_lens)
 
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()
 
        acc = evaluate(model, val_loader)
        print(f"Эпоха {epoch:2d}/{epochs}  loss={running / step:.3f}  "
              f"точность на валидации={acc:.1%}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)
 
    print(f"Готово. Лучшая точность: {best_acc:.1%}. Модель: {MODEL_PATH}")
 
 
# ----------------------------------------------------------------------------
# Предсказание
# ----------------------------------------------------------------------------
def predict(image_path: str) -> str:
    model = CRNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    x = image_to_tensor(Image.open(image_path)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return greedy_decode(model(x))[0]
 
 
def demo(path: str = "demo.png"):
    text = "".join(random.choices(CHARS, k=CAPTCHA_LEN))
    ImageCaptcha(width=W, height=H).write(text, path)
    print(f"Сохранено {path}, правильный ответ: {text}")
 
 
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
 
    p_train = sub.add_parser("train", help="обучить модель")
    p_train.add_argument("--epochs", type=int, default=15)
    p_train.add_argument("--samples", type=int, default=20000,
                         help="сколько капч генерировать за эпоху")
    p_train.add_argument("--batch", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
 
    p_pred = sub.add_parser("predict", help="распознать картинку")
    p_pred.add_argument("--image", required=True)
 
    sub.add_parser("demo", help="сгенерировать пример капчи demo.png")
 
    args = parser.parse_args()
    if args.cmd == "train":
        train(args.epochs, args.samples, args.batch, args.lr)
    elif args.cmd == "predict":
        print(predict(args.image))
    else:
        demo()
 