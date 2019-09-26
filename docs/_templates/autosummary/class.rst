{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :no-private-members:
   :no-inherited-members:

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :toctree: {{ objname }}
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
