if [-z "${ENERGYAI_WEB_PORT}"]
  then
    streamlit run web/web.py --server.address 0.0.0.0 --server.port 19999
  else
    streamlit run web/web.py --server.address 0.0.0.0 --server.port "${ENERGYAI_WEB_PORT}"
fi