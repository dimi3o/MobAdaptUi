from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QListWidget,QLineEdit, QTextEdit, QInputDialog, QHBoxLayout, QVBoxLayout, QFormLayout
import json

app = QApplication([])
'''Интерфейс приложения'''
notes_win = QWidget()
notes_win.setWindowTitle('Умные заметки')
notes_win.resize(900, 600)
list_notes = QListWidget()
list_notes_label = QLabel('Список заметок')
button_note_create = QPushButton('Создать заметку')
button_note_del = QPushButton('Удалить заметку')
button_choose = QPushButton('Выбрать заметку для соединения')
button_note_save = QPushButton('Сохранить заметку')
field_text = QTextEdit()
button_together = QPushButton('Соеденить заметки')
list_tags = QListWidget()
list_tags_label = QLabel('Список заметок для соединения')
layout_notes = QHBoxLayout()
col_1 = QVBoxLayout()
col_1.addWidget(field_text)
col_2 = QVBoxLayout()
col_2.addWidget(list_notes_label)
col_2.addWidget(list_notes)
row_1 = QHBoxLayout()
row_1.addWidget(button_note_create)
row_1.addWidget(button_note_del)
row_2 = QHBoxLayout()
row_2.addWidget(button_choose)
row_2.addWidget(button_note_save)
col_2.addLayout(row_1)
col_2.addLayout(row_2)
col_2.addWidget(list_tags_label)
col_2.addWidget(list_tags)
row_3 = QHBoxLayout()
row_3.addWidget(button_together)
row_4 = QHBoxLayout()
col_2.addLayout(row_3)
col_2.addLayout(row_4)
layout_notes.addLayout(col_1, stretch=2)
layout_notes.addLayout(col_2, stretch=1)
notes_win.setLayout(layout_notes)
notes_path='notes_data.json'
'''Функционал приложения'''


def show_note():
    key = list_notes.selectedItems()[0].text()
    print(key)
    field_text.setText(notes[key]["текст"])


def add_note():
    note_name, ok = QInputDialog.getText(notes_win, "Добавить заметку", "Название    заметки: ")
    if ok and note_name != "":
        notes[note_name] = {"текст": ""}
    list_notes.addItem(note_name)
    print(notes)

def save_note():
    if list_notes.selectedItems():
        key = list_notes.selectedItems()[0].text()
        notes[key]["текст"] = field_text.toPlainText()
        save_notes(notes)
        print(notes)
    else:
        print("Заметка для сохранения не выбрана!")

def del_note():
    if list_notes.selectedItems():
        key = list_notes.selectedItems()[0].text()
        del notes[key]
        list_notes.clear()
        field_text.clear()
        list_notes.addItems(notes)
        save_notes(notes)
        print(notes)
    else:
        print("Заметка для удаления не выбрана!")


def choose_note():
    global notes_to_unite
    if list_notes.selectedItems():
        key = list_notes.selectedItems()[0].text()
        list_tags.addItem(key)
        notes_to_unite.append(key)


def unite_notes():
    global notes_to_unite
    notes_txt = ''
    for key in notes_to_unite:
        notes_txt += notes[key]["текст"] + '\n'
    note_name, ok = QInputDialog.getText(notes_win, "Добавить заметку", "Название    заметки: ")
    if ok and note_name != "":
        notes[note_name] = {"текст": notes_txt}
        list_notes.addItem(note_name)
        print(notes)
        save_notes(notes)
    notes_to_unite = []

def save_notes(notes):
    with open(notes_path, "w", encoding='utf8') as file:
        json.dump(notes, file, sort_keys=True, ensure_ascii=False)

def load_notes():
    with open(notes_path, "r",encoding='utf8') as file:
        return json.load(file)

list_notes.itemClicked.connect(show_note)
button_note_create.clicked.connect(add_note)
button_note_save.clicked.connect(save_note)
button_choose.clicked.connect(choose_note)
button_together.clicked.connect(unite_notes)
button_note_del.clicked.connect(del_note)
list_notes.itemClicked.connect(show_note)

notes_to_unite = []
notes=load_notes()
notes_win.show()

with open("notes_data.json", "r") as file:
    notes = json.load(file)
list_notes.addItems(notes)

app.exec_()
