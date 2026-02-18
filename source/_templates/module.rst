{{ fullname }}
{{ '=' * (fullname|length) }}

.. currentmodule:: {{ module }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   
.. raw:: html

   <div class="module-short-summary">{{ module_doc | striptags | truncate(200) }}</div>
