# evaluator.py

import os
import argparse
import yaml
import json
import gc
import random
import torch
import numpy as np
import logging
import imageio

import sys
import json
import altair as alt
import pandas as pd
import cv2
from sentence_transformers import SentenceTransformer
from paddleocr import PaddleOCR
from tqdm import tqdm
import torch
import logging
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
# from ppocr.utils.logging import get_logger
from transformers import AutoImageProcessor, AutoTokenizer, BertModel, SwinModel
import pandas as pd
import json
from stable_baselines3 import PPO
import csv
import os
import glob
from copy import copy
import matplotlib.pyplot as plt

from chartist_case.model.salformer import SalFormer
from chartist_case.model.llm_policy import LLMPolicy
from chartist_case.analysis.taskvisdata.visualization_utils import plot_img, plot_scanpath

# Insert the chartist_case folder into sys.path so that "model" can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chartist_case"))

ocr = PaddleOCR(use_angle_cls=True, lang="en")  # need to run only once to download and load model into memory

ACTION = 20
IMAGE_SIZE = 320
READ_VALUE = 'retrieve_value'
FILTERING = 'filter'
FIND_EXTREME = 'find_extreme'

sentenceTransformer = SentenceTransformer("stsb-roberta-base-v2")
device = 'cpu'  # only works with cpu
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
vit = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

model = SalFormer(vit, bert).to(device)
checkpoint = torch.load('./chartist_case/outputs/VisSalFormer_weights.tar', map_location=torch.device(device))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --------------------------------------------------------------------------
# Set up a global logger reference for the entire file. We'll use the same
# name "EvaluatorChartist" that is set up in `setup_logger()` in main().
logger = logging.getLogger("EvaluatorChartist")
# --------------------------------------------------------------------------

def generate_chart_from_data1(file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value):
    logger.debug("Entering generate_chart_from_data1 with parameters: file_path=%s, bar_width=%s, aspect_ratio=%s, x_axis_font_size=%s, y_axis_font_size=%s, r_value=%s, g_value=%s, b_value=%s",
                 file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value)

    color_string = "rgb(%s,%s,%s)" % (r_value, g_value, b_value)

    # Sample data
    data = pd.DataFrame({
        'Theme Park': [
            'Magic Kingdom at Walt Disney World (US)', 'Disneyland (US)', 'Tokyo Disneyland (Japan)',
            'Tokyo Disney Sea (Japan)', 'Disneyland Park (France)', 'Epcot at Walt Disney World (US)',
            'Animal Kingdom at Walt Disney World (US)', 'Hollywood Studios at Walt Disney World (US)',
            'Universal Studios Japan (Japan)', 'Islands of Adventure at Universal Orlando (US)',
            'Ocean Park (Hong kong)', 'Everland (South Korea)', "Disney's California Adventure (US)",
            'Universal Studios (US)', 'Hong Kong Disneyland (Hong Kong)', 'Nagashima Spa Land (Japan)',
            'Lotte World (South Korea)', 'Seaworld Florida (US)', 'Universal Studios Hollywood (US)',
            'Walt Disney Studios Park (France)'
        ],
        'Attendance': [
            17.0, 16.1, 14.0, 11.9, 10.5, 10.5, 9.8, 9.7, 8.1, 7.7, 7.1, 6.9, 6.3, 6.0, 5.9, 5.8, 5.4, 5.1, 5.1, 4.7
        ],
        'Percent Change': [
            1.0, 1.0, -3.2, -5.8, 4.7, 'Nil', 1.0, 1.0, 4.2, 29.0, 28.7, -4.6, 1.0, 2.0, 13.5, 30.3, 4.1, 2.0, 2.0, 4.7
        ]
    })

    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Attendance:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 20])),
        y=alt.Y('Theme Park:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=300),
                sort=data['Theme Park'].tolist()),
        color=alt.value(color_string)
    )

    text = alt.Chart(data).mark_text(
        align='left',
        baseline='middle',
        dx=3,
        fontSize=12,
        color='black'
    ).encode(
        x=alt.value(250),
        y=alt.Y('Theme Park:N', sort=data['Theme Park'].tolist()),
    ).transform_filter(
        alt.datum['Percent Change'] != None
    )

    chart = (bars + text).properties(
        title={
            "text": "Top 20 theme parks worldwide",
            "subtitle": ["By attendance, 2011, m"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )

    chart.save(file_path + '/chart_design.png')
    logger.debug("Saved chart_design.png for data 1 to %s", file_path)

    # Additional versions (retrieve_value, filter, find_extreme)
    # retrieve_value
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Attendance:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 20])),
        y=alt.Y('Theme Park:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=300),
                sort=data['Theme Park'].tolist()),
        color=alt.condition(
            alt.datum['Theme Park'] == 'Universal Studios Hollywood (US)',
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Top 20 theme parks worldwide",
            "subtitle": ["By attendance, 2011, m"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )

    chart.save(file_path + '/chart_design_retrieve_value.png')
    logger.debug("Saved chart_design_retrieve_value.png for data 1")

    # filter
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Attendance:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 20])),
        y=alt.Y('Theme Park:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=300),
                sort=data['Theme Park'].tolist()),
        color=alt.condition(
            alt.datum['Theme Park'] == 'Magic Kingdom at Walt Disney World (US)',
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Top 20 theme parks worldwide",
            "subtitle": ["By attendance, 2011, m"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_filter.png')
    logger.debug("Saved chart_design_filter.png for data 1")

    # find_extreme
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Attendance:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 20])),
        y=alt.Y('Theme Park:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=300),
                sort=data['Theme Park'].tolist()),
        color=alt.condition(
            alt.datum['Theme Park'] == 'Magic Kingdom at Walt Disney World (US)',
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Top 20 theme parks worldwide",
            "subtitle": ["By attendance, 2011, m"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_find_extreme.png')
    logger.debug("Saved chart_design_find_extreme.png for data 1")
    logger.debug("Exiting generate_chart_from_data1")


def generate_chart_from_data2(file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value):
    logger.debug("Entering generate_chart_from_data2 with parameters: file_path=%s, bar_width=%s, aspect_ratio=%s, x_axis_font_size=%s, y_axis_font_size=%s, r_value=%s, g_value=%s, b_value=%s",
                 file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value)

    color_string = "rgb(%s,%s,%s)" % (r_value, g_value, b_value)

    category = [
        "Valdivia-Puerto Montt, Chile (1960)",
        "Prince William Sound, Alaska (1964)",
        "Northern Sumatra, Indonesia (2004)",
        "Kamchatka, Russia (1952)",
        "Coast of Honshu, Japan (2011)",
        "Coast of Maule, Chile (2010)",
        "Coast of Ecuador (1906)",
        "Rat Islands, Alaska (1965)",
        "Northern Sumatra, Indonesia (2005)",
        "Assam-Tibet border (1950)",
        "Andreanof Islands, Alaska (1957)",
        "Southern Sumatra, Indonesia (2007)",
        "Banda Sea, Indonesia (1938)",
        "Kamchatka, Russia (1923)",
        "Chile-Argentina Border (1922)",
        "Kurile Islands (1963)"
    ]

    value = [
        9.5, 9.2, 9.1, 9.0, 9.0, 8.8, 8.8, 8.7,
        8.6, 8.6, 8.6, 8.5, 8.5, 8.4, 8.3, 8.3
    ]

    df = pd.DataFrame({
        'category': category,
        'value': value
    })

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('value:Q', title='Magnitude', scale=alt.Scale(domain=[0, 10])),
        y=alt.Y('category:N', sort='-x', title=''),
        color=alt.value(color_string)
    ).properties(
        title={
            "text": "World's Largest Earthquakes",
            "subtitle": ["Since 1900, Magnitude"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=400 * aspect_ratio
    ).configure_axisX(
        labelFontSize=x_axis_font_size
    ).configure_axisY(
        labelFontSize=y_axis_font_size
    )

    chart.save(file_path + '/chart_design.png')
    logger.debug("Saved chart_design.png for data 2 to %s", file_path)

    # retrieve_value
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('value:Q', title='Magnitude', scale=alt.Scale(domain=[0, 10])),
        y=alt.Y('category:N', sort='-x', title=''),
        color=alt.condition(
            alt.datum['category'] == "Kamchatka, Russia (1952)",
            alt.value('black'),
            alt.value(color_string)
        )
    ).properties(
        title={
            "text": "World's Largest Earthquakes",
            "subtitle": ["Since 1900, Magnitude"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=400 * aspect_ratio
    ).configure_axisX(
        labelFontSize=x_axis_font_size
    ).configure_axisY(
        labelFontSize=y_axis_font_size
    )
    chart.save(file_path + '/chart_design_retrieve_value.png')
    logger.debug("Saved chart_design_retrieve_value.png for data 2")

    # filter
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('value:Q', title='Magnitude', scale=alt.Scale(domain=[0, 10])),
        y=alt.Y('category:N', sort='-x', title=''),
        color=alt.condition(
            alt.datum['category'] == "Coast of Honshu, Japan (2011)",
            alt.value('black'),
            alt.value(color_string)
        )
    ).properties(
        title={
            "text": "World's Largest Earthquakes",
            "subtitle": ["Since 1900, Magnitude"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=400 * aspect_ratio
    ).configure_axisX(
        labelFontSize=x_axis_font_size
    ).configure_axisY(
        labelFontSize=y_axis_font_size
    )
    chart.save(file_path + '/chart_design_filter.png')
    logger.debug("Saved chart_design_filter.png for data 2")

    # find_extreme
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('value:Q', title='Magnitude', scale=alt.Scale(domain=[0, 10])),
        y=alt.Y('category:N', sort='-x', title=''),
        color=alt.condition(
            alt.datum['category'] == "Valdivia-Puerto Montt, Chile (1960)",
            alt.value('black'),
            alt.value(color_string)
        )
    ).properties(
        title={
            "text": "World's Largest Earthquakes",
            "subtitle": ["Since 1900, Magnitude"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=300,
        height=400 * aspect_ratio
    ).configure_axisX(
        labelFontSize=x_axis_font_size
    ).configure_axisY(
        labelFontSize=y_axis_font_size
    )
    chart.save(file_path + '/chart_design_find_extreme.png')
    logger.debug("Saved chart_design_find_extreme.png for data 2")
    logger.debug("Exiting generate_chart_from_data2")


def generate_chart_from_data3(file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value):
    logger.debug("Entering generate_chart_from_data3 with parameters: file_path=%s, bar_width=%s, aspect_ratio=%s, x_axis_font_size=%s, y_axis_font_size=%s, r_value=%s, g_value=%s, b_value=%s",
                 file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value)

    color_string = "rgb(%s,%s,%s)" % (r_value, g_value, b_value)

    data = pd.DataFrame({
        'Country': [
            "Spain", "Portugal", "United States", "France", "Austria", "Italy", "Norway", "Britain",
            "Germany", "Canada", "Argentina", "Netherlands", "Australia", "Poland", "Sweden",
            "Brazil", "Greece", "Turkey", "Mexico"
        ],
        'Decreased': [
            32, 30, 25, 24, 23, 22, 21, 17, 16, 14, 14, 13.5, 13.3, 13, 12.5, 10, 4, 3.5, 2.5
        ]
    })

    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 50])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=100),
                sort=data['Country'].tolist()),
        color=alt.value(color_string)
    )

    text = alt.Chart(data).mark_text(
        align='left',
        baseline='middle',
        dx=3,
        fontSize=12,
        color='black'
    ).encode(
        x=alt.value(250),
        y=alt.Y('Country:N', sort=data['Country'].tolist()),
    )

    chart = (bars + text).properties(
        title={
            "text": "Organ donor rates",
            "subtitle": ["Per million population, 2010"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design.png')
    logger.debug("Saved chart_design.png for data 3 to %s", file_path)

    # retrieve_value
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 50])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=100),
                sort=data['Country'].tolist()),
        color=alt.condition(
            alt.datum['Country'] == "Brazil",
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Organ donor rates",
            "subtitle": ["Per million population, 2010"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_retrieve_value.png')
    logger.debug("Saved chart_design_retrieve_value.png for data 3")

    # filter
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 50])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=100),
                sort=data['Country'].tolist()),
        color=alt.condition(
            alt.datum['Country'] == "United States",
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Organ donor rates",
            "subtitle": ["Per million population, 2010"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_filter.png')
    logger.debug("Saved chart_design_filter.png for data 3")

    # find_extreme
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, orient='top'), scale=alt.Scale(domain=[0, 50])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=100),
                sort=data['Country'].tolist()),
        color=alt.condition(
            alt.datum['Country'] == "Mexico",
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Organ donor rates",
            "subtitle": ["Per million population, 2010"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_find_extreme.png')
    logger.debug("Saved chart_design_find_extreme.png for data 3")
    logger.debug("Exiting generate_chart_from_data3")


def generate_chart_from_data4(file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value):
    logger.debug("Entering generate_chart_from_data4 with parameters: file_path=%s, bar_width=%s, aspect_ratio=%s, x_axis_font_size=%s, y_axis_font_size=%s, r_value=%s, g_value=%s, b_value=%s",
                 file_path, bar_width, aspect_ratio, x_axis_font_size, y_axis_font_size, r_value, g_value, b_value)

    color_string = "rgb(%s,%s,%s)" % (r_value, g_value, b_value)

    data = pd.DataFrame({
        'Country': [
            "San Marino (1)",
            "Bermuda (2)",
            "Iceland (3)",
            "Ireland (8)",
            "Australia (11)",
            "Finland (=12)",
            "New Zealand (=12)",
            "Sweden (14)",
            "Great Britain (20)",
            "Canada (23)",
            "Ukraine (=31)",
            "Spain (=38)",
            "Cuba (=41)",
            "France (43)",
            "Germany (=52)",
            "Russia (=65)",
            "Rwanda (=69)",
            "Japan (=71)",
            "Iran (=77)",
            "Brazil (=81)",
            "Thailand (=89)",
            "United States (=89)",
            "Iraq (=93)",
            "China (=117)",
            "India (=152)",
            "Indonesia (=152)"
        ],
        'Decreased': [
            32, 15.5, 12.3, 10.5, 7.2, 7, 7, 6.5, 4.8, 4.5, 3.5, 3, 2.5, 2.5, 2.3, 2, 1.8, 1.6, 1.4, 1.2, 1.1, 1, 0.8, 0.6, 0.3, 0.2
        ]
    })

    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, tickCount=6, orient='top'),
                scale=alt.Scale(domain=[0, 36])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=150),
                sort=data['Country'].tolist()),
        color=alt.value(color_string)
    )

    text = alt.Chart(data).mark_text(
        align='left',
        baseline='middle',
        dx=3,
        fontSize=12,
        color='black'
    ).encode(
        x=alt.value(250),
        y=alt.Y('Country:N', sort=data['Country'].tolist()),
    )

    chart = (bars + text).properties(
        title={
            "text": "Paralympic competitors",
            "subtitle": ["Per million population, summer 2012 games"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design.png')
    logger.debug("Saved chart_design.png for data 4 to %s", file_path)

    # retrieve_value
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, tickCount=6, orient='top'),
                scale=alt.Scale(domain=[0, 36])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=150),
                sort=data['Country'].tolist()),
        color=alt.condition(
            alt.datum['Country'] == "France (43)",
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Paralympic competitors",
            "subtitle": ["Per million population, summer 2012 games"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_retrieve_value.png')
    logger.debug("Saved chart_design_retrieve_value.png for data 4")

    # filter
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, tickCount=6, orient='top'),
                scale=alt.Scale(domain=[0, 36])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=150),
                sort=data['Country'].tolist()),
        color=alt.condition(
            alt.datum['Country'] == "Ireland (8)",
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Paralympic competitors",
            "subtitle": ["Per million population, summer 2012 games"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_filter.png')
    logger.debug("Saved chart_design_filter.png for data 4")

    # find_extreme
    bars = alt.Chart(data).mark_bar(size=bar_width).encode(
        x=alt.X('Decreased:Q', axis=alt.Axis(title=None, tickCount=6, orient='top'),
                scale=alt.Scale(domain=[0, 36])),
        y=alt.Y('Country:N', axis=alt.Axis(title=None, labelLimit=400, labelAlign='left', labelPadding=150),
                sort=data['Country'].tolist()),
        color=alt.condition(
            alt.datum['Country'] == "San Marino (1)",
            alt.value('black'),
            alt.value(color_string)
        )
    )

    chart = (bars + text).properties(
        title={
            "text": "Paralympic competitors",
            "subtitle": ["Per million population, summer 2012 games"],
            "fontSize": 20,
            "subtitleFontSize": 14,
            "anchor": "start",
            "align": "left",
        },
        width=500,
        height=500 * aspect_ratio
    ).configure_title(
        fontSize=12,
        subtitleFontSize=12,
        anchor='start',
        align='left',
        offset=20
    ).configure_axis(
        grid=False,
        domain=False,
        tickSize=0
    ).configure_axisX(
        grid=True,
        labelFontSize=x_axis_font_size,
        gridColor='#dddddd',
    ).configure_axisY(
        labelFontSize=y_axis_font_size,
    ).configure_view(
        strokeOpacity=0
    )
    chart.save(file_path + '/chart_design_find_extreme.png')
    logger.debug("Saved chart_design_find_extreme.png for data 4")
    logger.debug("Exiting generate_chart_from_data4")


def predict(ques: str, chart_url: str):
    """
    Execute the prediction.

    Args:
        ques: a question string to feed into VisSalFormer
        chart_url: path to the chart image
    Returns:
        heatmap from VisSalFormer (np.array)
    """
    logger.debug("Entering predict with question='%s' and chart_url='%s'", ques, chart_url)
    # image = Image.open(chart_url).convert("RGB")

    image = Image.open(chart_url).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    # Convert back to an RGB img
    image = Image.fromarray(thresh).convert("RGB")

    img_pt = image_processor(image, return_tensors="pt").to(device)
    inputs = tokenizer(ques, return_tensors="pt").to(device)

    mask = model(img_pt['pixel_values'], inputs)
    mask = mask.detach().cpu().squeeze().numpy()
    heatmap = (mask * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (ACTION, ACTION))
    logger.debug("Exiting predict; returning heatmap of shape %s", heatmap.shape)
    return heatmap


def prepare_meta_data(file_path, task, question):
    logger.debug("Entering prepare_meta_data with file_path=%s, task=%s, question=%s", file_path, task, question)

    meta_data = []
    json_data = {
        "id": "design",
        "image": "design",
        "task": task,
        "chart_type": "a horizontal bar chart",
        "question": question,
        "text_positions": [],
    }

    img_data = Image.open(file_path + '/chart_design.png')
    save_path = file_path + '/chart_design_origin.png'
    img_data.save(save_path)
    logger.debug("Saved chart_design_origin.png from chart_design.png")

    grid_size = (img_data.size[1], img_data.size[0])

    title_map = np.zeros(grid_size)
    mark_map = np.zeros(grid_size)
    axis_map = np.zeros(grid_size)

    title_map = cv2.resize(title_map, (ACTION, ACTION))
    cv2.imwrite(file_path + '/chart_design_title.png', title_map)
    mark_map = cv2.resize(mark_map, (ACTION, ACTION))
    cv2.imwrite(file_path + '/chart_design_mark.png', mark_map)
    axis_map = cv2.resize(axis_map, (ACTION, ACTION))
    cv2.imwrite(file_path + '/chart_design_axis.png', axis_map)
    logger.debug("Saved chart_design_title.png, chart_design_mark.png, chart_design_axis.png")

    search_map = np.zeros(grid_size)
    map_map = np.zeros(grid_size)
    map_mark_map = np.zeros(grid_size)
    read_map = np.zeros(grid_size)
    task_aoi_map = np.zeros(grid_size)

    step_x = int(grid_size[0] / ACTION)
    step_y = int(grid_size[1] / ACTION)
    search_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    map_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    map_mark_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    read_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    task_aoi_action_map = np.zeros((ACTION, ACTION, 1), dtype=np.uint8)
    for i in range(ACTION):
        for j in range(ACTION):
            if np.sum(search_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                search_action_map[i][j] = 255
                task_aoi_action_map[i][j] = 255
            if np.sum(map_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                map_action_map[i][j] = 255
                task_aoi_action_map[i][j] = 255
            if np.sum(map_mark_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                map_mark_action_map[i][j] = 255
            if np.sum(read_map[int(step_x*i):int(step_x*(i+1)), int(step_y*j):int(step_y*(j+1))]) > 0:
                read_action_map[i][j] = 255
                task_aoi_action_map[i][j] = 255
    cv2.imwrite(file_path + '/chart_design_search.png', search_action_map)
    cv2.imwrite(file_path + '/chart_design_map.png', map_action_map)
    cv2.imwrite(file_path + '/chart_design_map_mark.png', map_mark_action_map)
    cv2.imwrite(file_path + '/chart_design_read.png', read_action_map)
    logger.debug("Saved chart_design_search.png, chart_design_map.png, chart_design_map_mark.png, chart_design_read.png")

    saliency_map = predict(question, file_path + '/chart_design_origin.png')
    cv2.imwrite(file_path + '/chart_design_saliency.png', saliency_map)
    logger.debug("Saved chart_design_saliency.png")

    origin_image = cv2.imread(file_path + '/chart_design_origin.png')
    
    # Convert to grayscale
    gray = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
    # Simple threshold
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    resized_chart = cv2.resize(origin_image, (IMAGE_SIZE, IMAGE_SIZE))
    cv2.imwrite(file_path + '/chart_design.png', resized_chart)
    peripheral = cv2.resize(origin_image, (ACTION, ACTION))
    cv2.imwrite(file_path + '/chart_design_p.png', peripheral)
    logger.debug("Saved chart_design.png (resized) and chart_design_p.png (peripheral)")

    paddle_ocr_result = ocr.ocr(thresh)
    semantic_map = np.zeros((grid_size[0], grid_size[1]), dtype=np.uint8)
    text_positions = []
    for line in paddle_ocr_result[0]:
        position = line[0]
        text = line[1][0]
        score = line[1][1]
        sentences = [
            question,
            text,
        ]
        sentence_embeddings = sentenceTransformer.encode(sentences)
        similarity = cosine_similarity(sentence_embeddings[0].reshape(1, -1), sentence_embeddings[1].reshape(1, -1))
        semantic_similarity = float(similarity[0][0])
        if semantic_similarity < 0:
            semantic_similarity = 0
        color = (semantic_similarity * 255)
        cv2.rectangle(semantic_map, (int(position[0][0]), int(position[0][1])),
                      (int(position[2][0]), int(position[2][1])), color, -1)
        position = [
            int(position[0][0] * IMAGE_SIZE / grid_size[1]),
            int(position[0][1] * IMAGE_SIZE / grid_size[0]),
            int(position[2][0] * IMAGE_SIZE / grid_size[1]),
            int(position[2][1] * IMAGE_SIZE / grid_size[0])
        ]
        text_positions.append({
            "position": position,
            "text": text,
            "ocr_score": score,
            "semantic_similarity": semantic_similarity
        })
    semantic_map = cv2.resize(semantic_map, (ACTION, ACTION))
    cv2.imwrite(file_path + '/chart_design_semantic.png', semantic_map)
    logger.debug("Saved chart_design_semantic.png")

    json_data['text_positions'] = text_positions
    meta_data.append(json_data)

    json.dump(json_data, open(file_path + '/chart_design.json', 'w'))
    with open(file_path + '/meta_case.json', 'w') as f:
        json.dump(meta_data, f)
    logger.debug("Saved chart_design.json and meta_case.json")
    logger.debug("Exiting prepare_meta_data")


def get_image_dimensions(image_path):
    logger.debug("Entering get_image_dimensions with image_path=%s", image_path)
    with Image.open(image_path) as img:
        width, height = img.size
    logger.debug("Exiting get_image_dimensions with width=%s, height=%s", width, height)
    return width, height


def gen_scanpath(file_path):
    logger.debug("Entering gen_scanpath with file_path=%s", file_path)
    meta_json = file_path + '/meta_case.json'
    folder = file_path
    meta_infos = json.load(open(meta_json))
    for meta_info in tqdm(meta_infos):
        logger.debug("Processing meta_info with question: %s", meta_info['question'])
        search_model = PPO.load("./chartist_case/outputs/vision_search_case.pt", device="auto")
        map_model = PPO.load("./chartist_case/outputs/vision_map_case.pt", device="auto")
        read_model = PPO.load("./chartist_case/outputs/vision_read_case.pt", device="auto")
        title_model = PPO.load("./chartist_case/outputs/vision_title_case.pt", device="auto")
        model = LLMPolicy(search_model=search_model, map_model=map_model, read_model=read_model, title_model=title_model)

        ORIGIN_WIDTH, ORIGIN_HEIGHT = get_image_dimensions(file_path + '/chart_design_origin.png')

        csv_path = os.path.join(file_path, 'design.csv')
        with open(csv_path, mode='w') as scanpath_file:
            scanpath_writer = csv.writer(scanpath_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            scanpath_writer.writerow(["user", "index", "x", "y"])

            # Just generating one "user" scenario here
            for i in tqdm(range(1)):
                model.setup(meta_info, meta_json=meta_json, folder=folder)
                step = 0
                done = False
                while not done:
                    step += 1
                    done = model.action()

                scanpath_xs = model.scanpath_xs
                scanpath_ys = model.scanpath_ys

                for j in range(len(scanpath_xs)):
                    x = int(scanpath_xs[j] * (ORIGIN_WIDTH / IMAGE_SIZE))
                    y = int(scanpath_ys[j] * (ORIGIN_HEIGHT / IMAGE_SIZE))
                    scanpath_writer.writerow([i, j, x, y])
        logger.debug("Wrote scanpath to %s", csv_path)
    logger.debug("Exiting gen_scanpath")

def extend_aoi_area(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image so that black areas become white (in the binary mask)
    # This threshold value (10) assumes pixels near 0 are black.
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)

    # Loop over the contours to find a black rectangle that is bigger than 20x20
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 2:
            # Get coordinates of the rectangle: (x1, y1) is top-left and (x2, y2) is bottom-right.
            x1, y1, x2, y2 = x, y, x + w, y + h
            print("Found black rectangle at:", (x1, y1, x2, y2))
            
            # Change the rectangle by extending it to start from x=0.
            # That is, fill the area from (0, y1) to (x2, y2) with black.
            img[y1:y2, 0:x2] = (0, 0, 0)
            
            # Process only the first valid rectangle found.
            break

    # Save the modified image
    cv2.imwrite(image_path, img)

def measure_aoi(file_path, task):
    logger.debug("Entering measure_aoi with file_path=%s and task=%s", file_path, task)
    
    # Read CSV file and log its shape.
    df = pd.read_csv(file_path + '/design.csv')
    logger.debug("DataFrame shape: %s", df.shape)
    
    # Load the original chart image and log its shape.
    chart = cv2.imread(file_path + '/chart_design_origin.png')
    if chart is not None:
        logger.debug("Chart origin shape: %s", chart.shape)
    else:
        logger.error("Chart origin image not found at %s", file_path + '/chart_design_origin.png')
        return 0  # or handle error
    
    # Load the task-specific AOI image and log its shape.
    if task == READ_VALUE:
        extend_aoi_area(file_path + '/chart_design_%s.png' % READ_VALUE)
        chart_aoi = cv2.imread(file_path + '/chart_design_%s.png' % READ_VALUE)
    elif task == FILTERING:
        extend_aoi_area(file_path + '/chart_design_%s.png' % FILTERING)
        chart_aoi = cv2.imread(file_path + '/chart_design_%s.png' % FILTERING)
    elif task == FIND_EXTREME:
        extend_aoi_area(file_path + '/chart_design_%s.png' % FIND_EXTREME)
        chart_aoi = cv2.imread(file_path + '/chart_design_%s.png' % FIND_EXTREME)
    else:
        chart_aoi = None

    if chart_aoi is not None:
        logger.debug("Chart AOI shape: %s", chart_aoi.shape)
    else:
        logger.error("Chart AOI image not found for task: %s", task)
        return 0  # or handle error
    
    # 1. Convert both images to grayscale
    chart_gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
    chart_aoi_gray = cv2.cvtColor(chart_aoi, cv2.COLOR_BGR2GRAY)
    
    # 2. Threshold them to binary
    _, chart_bin = cv2.threshold(chart_gray, 160, 255, cv2.THRESH_BINARY)
    _, chart_aoi_bin = cv2.threshold(chart_aoi_gray, 160, 255, cv2.THRESH_BINARY)

    # ---- Save the binary images for debugging ----
    chart_bin_path = file_path + '/chart_binary.png'
    chart_aoi_bin_path = file_path + '/chart_aoi_binary.png'
    cv2.imwrite(chart_bin_path, chart_bin)
    cv2.imwrite(chart_aoi_bin_path, chart_aoi_bin)
    logger.debug("Saved chart_binary.png at %s", chart_bin_path)
    logger.debug("Saved chart_aoi_binary.png at %s", chart_aoi_bin_path)
    # ---------------------------------------------
    
    # 3. Compute the difference
    #    Use absdiff to see absolute changes rather than directional difference.
    image_difference = cv2.absdiff(chart_aoi_bin, chart_bin)

    # 4. Save the difference image for debugging
    diff_file_path = file_path + '/chart_difference.png'
    cv2.imwrite(diff_file_path, image_difference)

    logger.debug("Difference image shape: %s", image_difference.shape)
    logger.debug("Difference image saved at %s", diff_file_path)
    
    # Copy x and y coordinates from the DataFrame and log their lengths.
    x = copy(df['y'].values)
    y = copy(df['x'].values)
    logger.debug("x coordinate array length: %d, y coordinate array length: %d", len(x), len(y))
    
    aoi_count = 0
    region_size = 30  # size of the region (15x15 pixels)
    half_region = region_size // 2
    
    # 5. Check each point in the difference image
    for i in range(len(x)):
        # Use x for row and y for column indices.
        row = x[i]
        col = y[i]
        
        # Compute bounding box for the region ensuring indices are within image bounds.
        row_start = max(row - half_region, 0)
        row_end = min(row + half_region + 1, image_difference.shape[0])
        col_start = max(col - half_region, 0)
        col_end = min(col + half_region + 1, image_difference.shape[1])
        
        region = image_difference[row_start:row_end, col_start:col_end]
        
        # If any pixel in the region is greater than 0, count it.
        if (region > 0).any():
            aoi_count += 1

    ratio = aoi_count / len(x) if len(x) else 0
    logger.debug("AOI count: %d out of %d points", aoi_count, len(x))
    logger.debug("Exiting measure_aoi with ratio=%.4f", ratio)
    return ratio


def gen_video(file_path):
    logger.debug("Entering gen_video with file_path=%s", file_path)

    # Read data
    df = pd.read_csv(file_path + '/design.csv')
    chart = Image.open(file_path + '/chart_design_origin.png')
    fig = plot_img(chart)  # Your custom function to set up a figure/plot

    # Extract coordinates
    x = copy(df['x'].values)
    y = copy(df['y'].values)

    # Generate and save frames
    for i in range(len(x)):
        fig = plot_img(chart)  # re-draw the chart

        frame_save_path = f"{file_path}/chart_design_video_{i:02d}.png"
        plt.savefig(frame_save_path)

        # Clean up the current figure to avoid overlap
        plt.close()

    # Collect frames
    image_files = sorted(glob.glob(file_path + '/chart_design_video_*.png'))
    if not image_files:
        logger.debug("No video frames found, exiting gen_video.")
        return

    # Create the video using imageio
    video_path = os.path.join(file_path, 'chart_design.mp4')
    fps = 2

    logger.debug("Writing video with imageio to %s", video_path)
    with imageio.get_writer(video_path, fps=fps) as writer:
        for img_file in image_files:
            image = imageio.imread(img_file)
            writer.append_data(image)

    # Remove the temporary PNG frames
    for file in image_files:
        os.remove(file)

    logger.debug("Exiting gen_video, video saved at %s", video_path)


def setup_logger():
    """
    Put the log file beside RESULT_FILE:
    run/<uid>/chartist_task/<mode>/<task>/results/trial_<n>/evaluator_chartist_<trial_id>.log
    Falls back to the old location if RESULT_FILE is missing.
    """
    trial_id = os.environ.get("TRIAL_ID", "unknown_trial")
    result_file = os.environ.get("RESULT_FILE", "")

    if result_file:
        trial_dir = os.path.dirname(result_file)
        log_dir = trial_dir
    else:
        # Fallback – stay consistent with orchestrator foldering
        base = os.path.abspath(os.getcwd())
        uid = os.environ.get("PARTICIPANT_ID", "anon")
        mode = os.environ.get("CHART_MODE", "tool")
        task = os.environ.get("CHART_TASK", "theme_parks")
        log_dir = os.path.join(base, "run", uid, "chartist_task", mode, task, "results", trial_id)

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"evaluator_chartist_{trial_id}.log")

    lg = logging.getLogger("EvaluatorChartist")
    lg.setLevel(logging.DEBUG)
    lg.propagate = False  # don't duplicate to root

    # (Re)create handlers each run so we always write to the right file
    for h in list(lg.handlers):
        lg.removeHandler(h)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    lg.addHandler(ch)
    lg.addHandler(fh)
    return lg


def set_random_seed(seed):
    logger.debug("Setting random seed to %s", seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    """
    Evaluator script:
    1) Reads config.yaml to get general training settings.
    2) Reads hyperparameters from HYPERPARAMS_FILE (set by orchestrator).
    3) Trains a RL model for the full number of timesteps with no intermediate eval.
    4) Evaluates the policy once at the end.
    5) Writes the final objective value to RESULT_FILE as JSON.
    """
    try:
        parser = argparse.ArgumentParser(description="Run a single Chartist job.")
        parser.add_argument("config_file_path", type=str, help="YAML config file path.")
        args = parser.parse_args()



        # 1) Load the main config.
        with open(args.config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        study_name = config["configuration"]["study_name"]
        chart_task = config["configuration"]["chart_task"]

        # Initialize logger
        global logger
        logger = setup_logger()

        logger.info(f"Chartist evaluator started with config file: {args.config_file_path}")
        logger.info("Main configuration loaded successfully.")

        # Log key environment variables and show the bash command to recreate the run.
        env_vars_used = {
            "HYPERPARAMS_FILE": os.environ.get("HYPERPARAMS_FILE", ""),
            "TRIAL_ID": os.environ.get("TRIAL_ID", ""),
            "RESULT_FILE": os.environ.get("RESULT_FILE", "")
        }
        logger.info("Environment variables used:\n%s", json.dumps(env_vars_used, indent=4))
        bash_command = (
            f"HYPERPARAMS_FILE='{env_vars_used['HYPERPARAMS_FILE']}' "
            f"TRIAL_ID='{env_vars_used['TRIAL_ID']}' "
            f"RESULT_FILE='{env_vars_used['RESULT_FILE']}' "
            f"python {os.path.basename(__file__)} {args.config_file_path}"
        )
        logger.info("To recreate this run, use the following bash command:\n%s", bash_command)

        # 2) Read hyperparams from JSON passed by orchestrator.
        hyperparams_file = os.environ.get("HYPERPARAMS_FILE", None)
        if hyperparams_file is None or not os.path.exists(hyperparams_file):
            logger.error("No valid HYPERPARAMS_FILE found in environment variables.")
            raise ValueError("No valid HYPERPARAMS_FILE found in environment variables.")

        with open(hyperparams_file, 'r') as hp_file:
            param_values = json.load(hp_file)
        logger.info(f"Hyperparameters loaded from file: {hyperparams_file}")

        seed = config["agent_training_configuration"]["random_seed"]
        if isinstance(seed, int):
            set_random_seed(seed)

        logger.info(f"Random seed set to: {seed}")

        trial_id = os.environ.get("TRIAL_ID", "unknown_trial")
        result_file = os.environ.get("RESULT_FILE", "")
        if result_file:
            trial_dir = os.path.dirname(result_file)  # .../results/trial_<n>
        else:
            # Fallback that mirrors orchestrator's run tree
            base = os.path.abspath(os.getcwd())
            uid = os.environ.get("PARTICIPANT_ID", "anon")
            mode = os.environ.get("CHART_MODE", "tool")
            task = os.environ.get("CHART_TASK", "theme_parks")
            trial_dir = os.path.join(base, "run", uid, "chartist_task", mode, task, "results", trial_id)

        files_path = os.path.join(trial_dir, "generated_charts")
        logger.info(f"Files path: {files_path}")
        os.makedirs(files_path, exist_ok=True)

        task = 'find_extreme'
        if chart_task == "paralympic":
            question = "Which country has most paralympic competitors per million population?"
            data_id = 4
        else:
            question = "Which theme park had the highest attendance in 2011?"
            data_id = 1

        design_params = {}
        for param_name, param_value in param_values.items():
            design_params[param_name] = param_value

        # Generate chart design
        if data_id == 1:
            generate_chart_from_data1(files_path, **design_params)
        elif data_id == 2:
            generate_chart_from_data2(files_path, **design_params)
        elif data_id == 3:
            generate_chart_from_data3(files_path, **design_params)
        elif data_id == 4:
            generate_chart_from_data4(files_path, **design_params)

        logger.info("Generated chart from data.")

        # Generate meta data
        prepare_meta_data(files_path, task, question)
        logger.info("Metadata prepared.")

        # Simulate scanpath
        gen_scanpath(files_path)
        logger.info("Scanpath generated.")

        # Evaluate once at the end
        obj_value = measure_aoi(files_path, task)
        extra_info = None
        logger.info(f"Model evaluation completed with objective value: {obj_value}")

        if extra_info is not None:
            extra_info_file = os.path.join(files_path, f"{trial_id}_eval_extra_info.json")
            with open(extra_info_file, "w") as fi:
                json.dump(extra_info, fi, indent=4)
            logger.info(f"Evaluation extra info saved to: {extra_info_file}")

        # Record video
        logger.info("Recording evaluation video(s)...")
        gen_video(files_path)

        # Write the final objective to RESULT_FILE
        result_file = os.environ.get("RESULT_FILE", None)
        if result_file is not None:
            with open(result_file, "w") as rf:
                json.dump({"objective_value": float(obj_value)}, rf)
            logger.info(f"Final objective value written to: {result_file}")
        else:
            logger.warning("RESULT_FILE environment variable not set; printing objective value instead.")
            print(json.dumps({"objective_value": float(obj_value)}))

        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleaned up model and memory.")

    except Exception as e:
        logger.exception("An error occurred during evaluation:")
        raise


if __name__ == "__main__":
    main()
