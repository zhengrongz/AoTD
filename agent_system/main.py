import warnings
warnings.filterwarnings("ignore")
import ast
import time
from rich.syntax import Syntax
from video_process import forward
from video_clip import *
from rich.console import Console
time_wait_between_lines = 0
console = Console(highlight=False, force_terminal=False)

def get_code(query, possible_answers=None, input_type="image"): 
    model_name = 'deepseek'
    code = forward(model_name, prompt=query, possible_answers=possible_answers, input_type=input_type)
    code = code.replace(f"execute_command({input_type})", f"execute_command({input_type}, time_wait_between_lines, syntax)")
    code = code.splitlines()
    if code:
        if 'python' in code[0]:
            code = code[1:-1]
        code[0] = code[0].lstrip()
    code = "\n".join(code)
    code_for_syntax = code.replace(f"({input_type}, time_wait_between_lines, syntax)", f"({input_type})")
    # syntax_1 = Syntax(code_for_syntax, "python", theme="monokai", line_numbers=True, start_line=0)
    # console.print(syntax_1)
    try:
        code = ast.unparse(ast.parse(code))
    except:
        return code, None
    code_for_syntax_2 = code.replace(f"({input_type}, time_wait_between_lines, syntax)", f"({input_type})")
    syntax_2 = Syntax(code_for_syntax_2, "python", theme="monokai", line_numbers=True, start_line=0)
    return code, syntax_2

def CodexAtLine(lineno, syntax, time_wait_between_lines=1.):
    syntax._stylized_ranges = []
    syntax.stylize_range('on red', (lineno + 1, 0), (lineno + 1, 80))
    time.sleep(time_wait_between_lines)


def get_thing_to_show_codetype(codeline):
    # can output either a list of things to show, or a single thing to show
    things_to_show = []
    if codeline.startswith("if"):
        condition, rest = codeline[3:].split(":", 1)
        codeline = f"if {condition}:{rest}"
        code_type = "if"

        operators = ['==', '!=', '>=', '<=', '>', '<']
        things_to_show = []
        for op in operators:
            if op in condition:
                things_to_show = [x.strip() for x in condition.split(op)]
                # print(things_to_show)
                break
        # things_to_show.append(thing_to_show)
        thing_to_show = things_to_show + [condition.strip()]

    elif codeline.startswith("for "):
        code_type = 'for'
        thing_to_show = codeline.split("for ")[1].split(" in ")[0]

    elif codeline.startswith("return"):
        thing_to_show = codeline.split("return ")[1]
        code_type = 'return'

    elif ' = ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' = ')[0]
    elif ' += ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' += ')[0]
    elif ' -= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' -= ')[0]
    elif ' *= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' *= ')[0]
    elif ' /= ' in codeline:
        code_type = 'assign'
        thing_to_show = codeline.split(' /= ')[0]

    elif '.append(' in codeline:
        code_type = 'append'
        thing_to_show = codeline.split('.append(')[0] + '[-1]'
    elif '.add(' in codeline:
        code_type = 'add'
        thing_to_show = codeline.split('.add(')[0]

    elif '.sort(' in codeline:
        code_type = 'sort'
        thing_to_show = codeline.split('.sort(')[0]

    elif codeline.startswith("elif") or codeline.startswith("else"):
        thing_to_show = None
        code_type = 'elif_else'
    else:
        thing_to_show = None
        code_type = 'other'

    if isinstance(thing_to_show, list):
        thing_to_show = [thing if not (thing.strip().startswith("'") and thing.strip().endswith("'"))
                         else thing.replace("'", '"') for thing in thing_to_show if thing is not None]
    elif isinstance(thing_to_show, str):
        thing_to_show = thing_to_show if not (thing_to_show.strip().startswith("'") and
                                              thing_to_show.strip().endswith("'")) else thing_to_show.replace("'", '"')
    return thing_to_show, code_type


def split_codeline_and_indent_level(codeline):
    origlen = len(codeline)
    codeline = codeline.lstrip()
    indent = origlen - len(codeline)
    indent = " " * indent
    return codeline, indent

def inject_saver(code, show_intermediate_steps, syntax=None, time_wait_between_lines=None, console=None):
    injected_function_name = 'show_all'
    if injected_function_name in code:
        return code
    code = code.split("\n")
    newcode = []
    first_indent = 0
    for n, codeline in enumerate(code):
        codeline, indent = split_codeline_and_indent_level(codeline)
        
        if codeline.startswith('#') or codeline == '':  # this will cause issues if you have lots of comment lines
            continue
        if '#' in codeline:
            codeline = codeline.split('#')[0]

        thing_to_show, code_type = get_thing_to_show_codetype(codeline)

        if code_type in ('assign', 'append', 'if', 'return', 'for', 'sort', 'add'):
            if '\'' in codeline:
                codeline.replace('\'', '\\\'')

            if show_intermediate_steps:
                escape_thing = lambda x: x.replace("'", "\\'")
                injection_string_format = \
                    lambda \
                        thing: f"{indent}{injected_function_name}(lineno={n},value=({thing}),valuename='{escape_thing(thing)}'," \
                               f"fig=my_fig,console_in=console,time_wait_between_lines=time_wait_between_lines); " \
                               f"CodexAtLine({n},syntax=syntax,time_wait_between_lines=time_wait_between_lines)"
            else:
                injection_string_format = lambda thing: f"{indent}CodexAtLine({n},syntax=syntax," \
                                                        f"time_wait_between_lines=time_wait_between_lines)"

            extension_list = []
            if isinstance(thing_to_show, list):
                injection_string_list = [injection_string_format(f"{thing}") for thing in thing_to_show]
                extension_list.extend(injection_string_list)
            elif code_type == 'for':
                injection_string = injection_string_format(f"{thing_to_show}")
                injection_string = " " * 4 + injection_string
                extension_list.append(injection_string)
            else:
                extension_list.append(injection_string_format(f"{thing_to_show}"))

            if code_type in ('if', 'return'):
                extension_list = extension_list + [f"{indent}{codeline}"]
            else:
                extension_list = [f"{indent}{codeline}"] + extension_list

            newcode.extend(extension_list)

        elif code_type == 'elif_else':
            newcode.append(f"{indent}{codeline}")
        else:
            newcode.append(f"{indent}{codeline}")
    return "\n".join(newcode)

def execute_code(code, im, query=None, possible_answers=None, show_intermediate_steps=False):
    code, syntax = code
    # print(code)
    code_line = inject_saver(code, show_intermediate_steps, syntax, time_wait_between_lines)
    exec(compile(code_line, 'Codex', 'exec'), globals())
    if query is None and possible_answers is None:
        result = execute_command(im, time_wait_between_lines, syntax)
    else:
        result = execute_command(im, query, possible_answers, time_wait_between_lines, syntax)  # The code is created in the exec()
    return result


